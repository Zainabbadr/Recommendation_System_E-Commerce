import os
import sys
import django
from pathlib import Path
import warnings
import sqlite3
import time
import uuid

# Suppress timezone warnings
warnings.filterwarnings('ignore', message='.*received a naive datetime.*')

# Fix the path and settings for your current project
sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendation_frontend.settings')

def create_new_database():
    """
    Create a new database with a different name to avoid corruption issues
    """
    from django.conf import settings
    from django.core.management import call_command
    
    original_db_path = settings.DATABASES['default']['NAME']
    
    # Create new database name
    db_dir = os.path.dirname(original_db_path)
    new_db_name = f"db_fresh_{int(time.time())}.sqlite3"
    new_db_path = os.path.join(db_dir, new_db_name)
    
    print(f"🔧 Creating new database: {new_db_path}")
    
    # Update Django settings to use new database
    settings.DATABASES['default']['NAME'] = new_db_path
    
    # Close any existing connections
    from django.db import connections
    connections.close_all()
    
    # Create fresh database with migrations
    try:
        print("🔨 Running migrations to create fresh database...")
        call_command('migrate', verbosity=1)
        print("✅ Fresh database created successfully")
        
        # Test the new database
        conn = sqlite3.connect(new_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM sqlite_master")
        conn.close()
        print("✅ New database is working correctly")
        
        return True
    except Exception as e:
        print(f"❌ Failed to create new database: {e}")
        return False

# Initialize Django
django.setup()

# Create new database
print("ℹ️ Using existing database (no new DB will be created).")


# Now import models after Django is set up and database is fixed
from recommendations.models import Dim_Products, Dim_Customers, Fact_Transactions
import csv
from django.utils.dateparse import parse_datetime
from decimal import Decimal

# [Rest of your functions remain the same - load_products, load_customers, load_all_transactions_with_error_reporting, load_data_to_db]

def load_products():
    """Load products from CSV file."""
    print("🛍️ Loading products...")
    
    base_path = Path(__file__).parent
    products_file = base_path / 'Dim_Products.csv'
    
    if not products_file.exists():
        print(f"❌ Products file not found: {products_file}")
        return False
    
    products_created = 0
    error_count = 0
    
    try:
        with open(products_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            products_to_create = []
            
            for row in reader:
                try:
                    product = Dim_Products(
                        StockCode=row['StockCode'],
                        Description=row.get('Description', ''),
                        # UnitPrice=Decimal(str(float(row['UnitPrice']))) if row.get('UnitPrice') else Decimal('0.00')
                    )
                    products_to_create.append(product)
                    
                    # Bulk create in batches
                    if len(products_to_create) >= 1000:
                        Dim_Products.objects.bulk_create(products_to_create, ignore_conflicts=True)
                        products_created += len(products_to_create)
                        products_to_create = []
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"⚠️ Product error: {e}")
            
            # Create remaining products
            if products_to_create:
                Dim_Products.objects.bulk_create(products_to_create, ignore_conflicts=True)
                products_created += len(products_to_create)
        
        final_count = Dim_Products.objects.count()
        print(f"✅ Products loaded: {final_count:,} (attempted: {products_created:,}, errors: {error_count})")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load products: {e}")
        return False


def load_customers():
    """Load customers (with Segment, District, and Customer_TotalSpending) from CSV file."""
    print("👥 Loading customers...")
    
    base_path = Path(__file__).parent
    customers_file = base_path / 'Dim_Customers.csv'
    
    if not customers_file.exists():
        print(f"❌ Customers file not found: {customers_file}")
        return False
    
    customers_created = 0
    error_count = 0
    
    try:
        with open(customers_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                # Normalize keys to lowercase to avoid case mismatch
                row = {k.strip().lower(): v for k, v in row.items()}
                
                try:
                    customer_id = int(float(row['customerid']))
                    country = row.get('country', '')
                    district = row.get('district', '')
                    segment = row.get('segment', '')  # lowercase from CSV
                    spending = Decimal(str(float(row['customer_totalspending']))) if row.get('customer_totalspending') else Decimal('0.00')

                    # Update if exists, otherwise create
                    Dim_Customers.objects.update_or_create(
                        CustomerID=customer_id,
                        defaults={
                            "Country": country,
                            "District": district,
                            "Segment": segment,
                            "Customer_TotalSpending": spending,
                        }
                    )
                    customers_created += 1
                
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"⚠️ Customer error: {e}")
        
        final_count = Dim_Customers.objects.count()
        print(f"✅ Customers loaded/updated: {final_count:,} (processed: {customers_created:,}, errors: {error_count})")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load customers: {e}")
        return False




def load_all_transactions_with_error_reporting():
    """Load ALL transactions with detailed error reporting."""
    print("🚀 Loading ALL transactions with detailed error reporting...")
    
    base_path = Path(__file__).parent
    transactions_file = base_path / 'Fact_Transactions.csv'
    
    if not transactions_file.exists():
        print(f"❌ Transactions file not found: {transactions_file}")
        return False
    
    print("💰 Loading ALL transactions...")
    
    transactions_to_create = []
    transactions_processed = 0
    transactions_saved = 0
    error_count = 0
    batch_size = 100
    
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
                        if error_count <= 10:
                            print(f"⚠️ Row {transactions_processed}: Foreign key error - {e}")
                        continue

                    # Parse data with better error handling
                    try:
                        quantity = int(row['Quantity']) if row.get('Quantity') else 0
                        unit_price = Decimal(str(float(row['UnitPrice']))) if row.get('UnitPrice') else Decimal('0.00')
                        total_price = Decimal(str(float(row['TotalPrice']))) if row.get('TotalPrice') else Decimal('0.00')
                    except (ValueError, TypeError) as e:
                        error_count += 1
                        if error_count <= 10:
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
                    
                    # Bulk create in batches
                    if len(transactions_to_create) >= batch_size:
                        try:
                            created_objects = Fact_Transactions.objects.bulk_create(transactions_to_create)
                            transactions_saved += len(created_objects)
                            transactions_to_create = []
                        except Exception as bulk_error:
                            print(f"❌ Bulk create error at transaction {transactions_processed}: {bulk_error}")
                            for trans in transactions_to_create:
                                try:
                                    trans.save()
                                    transactions_saved += 1
                                except Exception:
                                    error_count += 1
                            transactions_to_create = []
                    
                    # Progress tracking
                    if transactions_processed % 10000 == 0:
                        print(f"📈 Progress: {transactions_processed:,} processed, {transactions_saved:,} saved, {error_count} errors")
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"⚠️ Row {transactions_processed}: General error - {e}")
                    continue
            
            # Create remaining transactions
            if transactions_to_create:
                try:
                    created_objects = Fact_Transactions.objects.bulk_create(transactions_to_create)
                    transactions_saved += len(created_objects)
                except Exception as bulk_error:
                    print(f"❌ Final bulk create error: {bulk_error}")
                    for trans in transactions_to_create:
                        try:
                            trans.save()
                            transactions_saved += 1
                        except Exception:
                            error_count += 1
        
        # Verify actual database count
        final_count = Fact_Transactions.objects.count()
        print(f"\n✅ Loading completed!")
        print(f"📊 Final Summary:")
        print(f"  Total rows processed: {transactions_processed:,}")
        print(f"  Transactions saved: {transactions_saved:,}")
        print(f"  Actual database count: {final_count:,}")
        print(f"  Errors encountered: {error_count:,}")
        
        success_rate = (transactions_saved / transactions_processed) * 100 if transactions_processed > 0 else 0
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
    
    # Step 3: Load transactions
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
    # try:
    #     load_data_to_db()
    # except Exception as e:
    #     print(f"❌ Data loading failed with error: {e}")
    #     sys.exit(1)
    load_data_to_db()