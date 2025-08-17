import os
import sys
import django
from pathlib import Path
import warnings

# Suppress timezone warnings
warnings.filterwarnings('ignore', message='.*received a naive datetime.*')

# Fix the path and settings for your current project
sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendation_frontend.settings')
django.setup()

from recommendations.models import Dim_Products, Dim_Customers, Fact_Transactions
import csv
from django.utils.dateparse import parse_datetime
from decimal import Decimal


def load_all_transactions_with_error_reporting():
    """Load ALL transactions with detailed error reporting."""
    print("🚀 Loading ALL transactions with detailed error reporting...")
    
    base_path = Path(__file__).parent
    transactions_file = base_path / 'Fact_Transactions.csv'
    
    if not transactions_file.exists():
        print(f"❌ Transactions file not found: {transactions_file}")
        return False
    
    # Clear existing transaction data
    print("🗑️ Clearing existing transactions...")
    Fact_Transactions.objects.all().delete()
    
    print("💰 Loading ALL transactions...")
    
    transactions_to_create = []
    transactions_processed = 0
    transactions_saved = 0
    error_count = 0
    batch_size = 1000  # Smaller batch size for better error tracking
    
    try:
        with open(transactions_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                transactions_processed += 1
                
                try:
                    # Get foreign key objects
                    customer_id = int(float(row['CustomerID']))
                    stock_code = row['StockCode']
                    
                    try:
                        customer = Dim_Customers.objects.get(CustomerID=customer_id)
                        product = Dim_Products.objects.get(StockCode=stock_code)
                    except (Dim_Customers.DoesNotExist, Dim_Products.DoesNotExist) as e:
                        error_count += 1
                        if error_count <= 10:  # Show first 10 FK errors
                            print(f"⚠️ Row {transactions_processed}: Foreign key error - {e}")
                        continue

                    # Parse data with better error handling
                    try:
                        quantity = int(row['Quantity']) if row.get('Quantity') else 0
                        unit_price = Decimal(str(float(row['UnitPrice']))) if row.get('UnitPrice') else Decimal('0.00')
                        total_price = Decimal(str(float(row['TotalPrice']))) if row.get('TotalPrice') else Decimal('0.00')
                    except (ValueError, TypeError) as e:
                        error_count += 1
                        if error_count <= 10:  # Show first 10 parsing errors
                            print(f"⚠️ Row {transactions_processed}: Data parsing error - {e}")
                        continue
                    
                    # Handle date
                    invoice_date = None
                    if row.get('InvoiceDate'):
                        invoice_date = parse_datetime(row['InvoiceDate'])

                    # Create transaction object
                    transaction_obj = Fact_Transactions(
                        InvoiceNo=row['InvoiceNo'],
                        CustomerID=customer,
                        StockCode=product,
                        Quantity=quantity,
                        UnitPrice=unit_price,
                        TotalPrice=total_price,
                        InvoiceDate=invoice_date,
                    )
                    
                    transactions_to_create.append(transaction_obj)
                    
                    # Bulk create in batches WITHOUT ignore_conflicts
                    if len(transactions_to_create) >= batch_size:
                        try:
                            # Remove ignore_conflicts to see actual errors
                            created_objects = Fact_Transactions.objects.bulk_create(transactions_to_create)
                            transactions_saved += len(created_objects)
                            transactions_to_create = []
                        except Exception as bulk_error:
                            print(f"❌ Bulk create error at transaction {transactions_processed}: {bulk_error}")
                            # Try individual creates to identify specific problems
                            for i, trans in enumerate(transactions_to_create):
                                try:
                                    trans.save()
                                    transactions_saved += 1
                                except Exception as individual_error:
                                    error_count += 1
                                    if error_count <= 10:  # Show first 10 individual errors
                                        print(f"⚠️ Individual save error: {individual_error}")
                            transactions_to_create = []
                    
                    # Progress tracking every 10,000 transactions
                    if transactions_processed % 10000 == 0:
                        print(f"📈 Progress: {transactions_processed:,} processed, {transactions_saved:,} saved, {error_count} errors")
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:  # Show first 10 general errors
                        print(f"⚠️ Row {transactions_processed}: General error - {e}")
                    continue
            
            # Create remaining transactions
            if transactions_to_create:
                try:
                    created_objects = Fact_Transactions.objects.bulk_create(transactions_to_create)
                    transactions_saved += len(created_objects)
                except Exception as bulk_error:
                    print(f"❌ Final bulk create error: {bulk_error}")
                    # Try individual creates
                    for trans in transactions_to_create:
                        try:
                            trans.save()
                            transactions_saved += 1
                        except Exception:
                            error_count += 1
        
        # Final progress report
        print(f"📈 Final: {transactions_processed:,} processed, {transactions_saved:,} saved, {error_count} errors")
        
        # Verify actual database count
        final_count = Fact_Transactions.objects.count()
        print(f"\n✅ Loading completed!")
        print(f"📊 Final Summary:")
        print(f"  Total rows processed: {transactions_processed:,}")
        print(f"  Transactions saved: {transactions_saved:,}")
        print(f"  Actual database count: {final_count:,}")
        print(f"  Errors encountered: {error_count:,}")
        
        success_rate = (transactions_saved / transactions_processed) * 100
        print(f"  Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Critical error during loading: {e}")
        return False
    
    return True

def load_data_to_db():
    """Load all data to database in the correct order."""
    print("🚀 Starting comprehensive data loading...")
    
    # Step 1: Load products first
    if not load_products():
        print("❌ Failed to load products")
        return False
    
    # Step 2: Load customers
    if not load_customers():
        print("❌ Failed to load customers")
        return False
    
    # Step 3: Load transactions (depends on products and customers)
    if not load_all_transactions_with_error_reporting():
        print("❌ Failed to load transactions")
        return False
    
    print("✅ All data loaded successfully!")
    
    # Print final counts
    product_count = Dim_Products.objects.count()
    customer_count = Dim_Customers.objects.count()
    transaction_count = Fact_Transactions.objects.count()
    
    print(f"📊 Final Database Counts:")
    print(f"  Products: {product_count:,}")
    print(f"  Customers: {customer_count:,}")
    print(f"  Transactions: {transaction_count:,}")
    
    return True

if __name__ == "__main__":
    load_data_to_db()