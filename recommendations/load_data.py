import os
import sys
import django
from pathlib import Path

# Fix the path and settings for your current project
sys.path.append(str(Path(__file__).parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'recommendation_frontend.settings')
django.setup()

from recommendations.models import Dim_Products, Dim_Customers, Fact_Transactions
import csv
from django.utils.dateparse import parse_datetime
from decimal import Decimal

def load_csv_data():
    """Load data from CSV files with proper error handling."""
    print("🚀 Starting CSV to SQLite migration...")
    
    # Define file paths using raw strings or Path
    base_path = Path(__file__).parent
    csv_files = {
        'products': base_path / 'Dim_Products.csv',
        'customers': base_path / 'Dim_Customers.csv', 
        'transactions': base_path / 'Fact_Transactions.csv'
    }
    
    # Clear existing data (optional)
    print("🗑️ Clearing existing data...")
    Fact_Transactions.objects.all().delete()
    Dim_Products.objects.all().delete()
    Dim_Customers.objects.all().delete()
    
    # 1. Load Dim_Products
    print("📦 Loading products...")
    try:
        with open(csv_files['products'], newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            products_count = 0
            for row in reader:
                Dim_Products.objects.update_or_create(
                    StockCode=row['StockCode'],
                    defaults={
                        'Description': (row.get('Description') or None)[:50] if row.get('Description') else None,
                        'Description_Categorize': (row.get('Description_Categorize') or None)[:50] if row.get('Description_Categorize') else None,
                    }
                )
                products_count += 1
        print(f"✅ Created {products_count} products")
    except FileNotFoundError:
        print(f"❌ Products file not found: {csv_files['products']}")
        return False
    except Exception as e:
        print(f"❌ Error loading products: {e}")
        return False

    # 2. Load Dim_Customers
    print("👥 Loading customers...")
    try:
        with open(csv_files['customers'], newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            customers_count = 0
            for row in reader:
                try:
                    # Handle Customer_TotalSpending conversion
                    total_spending = row.get('Customer_TotalSpending', 0)
                    if total_spending and total_spending != '0':
                        total_spending = Decimal(str(float(total_spending)))
                    else:
                        total_spending = Decimal('0.00')
                    
                    Dim_Customers.objects.update_or_create(
                        CustomerID=int(float(row['CustomerID'])),
                        defaults={
                            'Country': (row.get('Country') or None)[:100] if row.get('Country') else None,
                            'District': (row.get('District') or None)[:100] if row.get('District') else None,
                            'Customer_TotalSpending': total_spending,
                            'Segment': (row.get('Segment') or None)[:100] if row.get('Segment') else None,
                        }
                    )
                    customers_count += 1
                except ValueError as e:
                    print(f"⚠️ Skipping customer row due to data error: {e}")
                    continue
        print(f"✅ Created {customers_count} customers")
    except FileNotFoundError:
        print(f"❌ Customers file not found: {csv_files['customers']}")
        return False
    except Exception as e:
        print(f"❌ Error loading customers: {e}")
        return False

    # 3. Load Fact_Transactions
    print("💰 Loading transactions...")
    try:
        with open(csv_files['transactions'], newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            transactions_count = 0
            error_count = 0
            max_transactions = 20000
            
            for row in reader:
                # Stop if we've reached the limit
                if transactions_count >= max_transactions:
                    print(f"🔢 Reached limit of {max_transactions} transactions, stopping...")
                    break
                try:
                    # Get foreign key objects with better error handling
                    customer_id = int(float(row['CustomerID']))
                    stock_code = row['StockCode']
                    
                    try:
                        customer = Dim_Customers.objects.get(CustomerID=customer_id)
                        product = Dim_Products.objects.get(StockCode=stock_code)
                    except Dim_Customers.DoesNotExist:
                        print(f"⚠️ Customer {customer_id} not found for invoice {row.get('InvoiceNo', 'Unknown')}")
                        error_count += 1
                        continue
                    except Dim_Products.DoesNotExist:
                        print(f"⚠️ Product {stock_code} not found for invoice {row.get('InvoiceNo', 'Unknown')}")
                        error_count += 1
                        continue

                    # Handle numeric conversions with Decimal for precision
                    quantity = int(row['Quantity']) if row.get('Quantity') else 0
                    unit_price = Decimal(str(float(row['UnitPrice']))) if row.get('UnitPrice') else Decimal('0.00')
                    total_price = Decimal(str(float(row['TotalPrice']))) if row.get('TotalPrice') else Decimal('0.00')
                    
                    # Handle date parsing
                    invoice_date = None
                    if row.get('InvoiceDate'):
                        invoice_date = parse_datetime(row['InvoiceDate'])
                        if not invoice_date:
                            # Try alternative date parsing if needed
                            from datetime import datetime
                            try:
                                invoice_date = datetime.strptime(row['InvoiceDate'], '%m/%d/%Y %H:%M')
                            except ValueError:
                                print(f"⚠️ Invalid date format for invoice {row.get('InvoiceNo', 'Unknown')}: {row['InvoiceDate']}")

                    Fact_Transactions.objects.update_or_create(
                        InvoiceNo=row['InvoiceNo'],
                        defaults={
                            'CustomerID': customer,
                            'StockCode': product,
                            'Quantity': quantity,
                            'UnitPrice': unit_price,
                            'TotalPrice': total_price,
                            'InvoiceDate': invoice_date,
                        }
                    )
                    transactions_count += 1
                    
                    # Progress indicator
                    if transactions_count % 1000 == 0:
                        print(f"📈 Processed {transactions_count} transactions...")
                        
                except (ValueError, KeyError) as e:
                    print(f"⚠️ Skipping transaction row due to data error: {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    print(f"⚠️ Unexpected error processing transaction: {e}")
                    error_count += 1
                    continue
                    
        print(f"✅ Created {transactions_count} transactions")
        if error_count > 0:
            print(f"⚠️ Skipped {error_count} rows due to errors")
            
    except FileNotFoundError:
        print(f"❌ Transactions file not found: {csv_files['transactions']}")
        return False
    except Exception as e:
        print(f"❌ Error loading transactions: {e}")
        return False
    
    # Print final summary
    print("\n📊 Migration Summary:")
    print(f"Products: {Dim_Products.objects.count()}")
    print(f"Customers: {Dim_Customers.objects.count()}")
    print(f"Transactions: {Fact_Transactions.objects.count()}")
    
    print("🎉 Migration completed successfully!")
    return True


def continue_loading_transactions_from_invoice(start_after_invoice="573167", max_transactions=None):
    """Continue loading transactions from after a specific invoice number to the end."""
    print(f"🚀 Continuing CSV to SQLite migration from after invoice {start_after_invoice} to end...")
    
    # Define file paths
    base_path = Path(__file__).parent
    transactions_file = base_path / 'Fact_Transactions.csv'
    
    # Check current database state
    current_transactions = Fact_Transactions.objects.count()
    current_products = Dim_Products.objects.count()
    current_customers = Dim_Customers.objects.count()
    
    print(f"📊 Current database state:")
    print(f"  Products: {current_products}")
    print(f"  Customers: {current_customers}")
    print(f"  Transactions: {current_transactions}")
    
    # Load Fact_Transactions starting from after specific invoice
    print(f"💰 Loading transactions starting after invoice {start_after_invoice} to end...")
    try:
        with open(transactions_file, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            transactions_count = 0
            error_count = 0
            row_count = 0
            skipped_count = 0
            found_start = False
            
            for row in reader:
                row_count += 1
                invoice_no = row.get('InvoiceNo', '')
                
                # Look for the starting invoice
                if not found_start:
                    if invoice_no == start_after_invoice:
                        found_start = True
                        print(f"🎯 Found starting invoice {start_after_invoice} at row {row_count}, continuing from next transaction...")
                    continue
                
                # Progress indicator for rows processed
                if row_count % 5000 == 0:
                    print(f"🔄 Processing row {row_count}... (Added: {transactions_count}, Skipped: {skipped_count}, Errors: {error_count})")
                
                try:
                    # Get foreign key objects with better error handling
                    customer_id = int(float(row['CustomerID']))
                    stock_code = row['StockCode']
                    
                    # Check if transaction already exists
                    if Fact_Transactions.objects.filter(InvoiceNo=invoice_no).exists():
                        skipped_count += 1
                        # Only print every 1000th skip to reduce output
                        if skipped_count % 1000 == 0:
                            print(f"⚠️ Skipped {skipped_count} existing transactions so far...")
                        continue
                    
                    try:
                        customer = Dim_Customers.objects.get(CustomerID=customer_id)
                        product = Dim_Products.objects.get(StockCode=stock_code)
                    except Dim_Customers.DoesNotExist:
                        print(f"⚠️ Customer {customer_id} not found for invoice {invoice_no}")
                        error_count += 1
                        continue
                    except Dim_Products.DoesNotExist:
                        print(f"⚠️ Product {stock_code} not found for invoice {invoice_no}")
                        error_count += 1
                        continue

                    # Handle numeric conversions with Decimal for precision
                    quantity = int(row['Quantity']) if row.get('Quantity') else 0
                    unit_price = Decimal(str(float(row['UnitPrice']))) if row.get('UnitPrice') else Decimal('0.00')
                    total_price = Decimal(str(float(row['TotalPrice']))) if row.get('TotalPrice') else Decimal('0.00')
                    
                    # Handle date parsing
                    invoice_date = None
                    if row.get('InvoiceDate'):
                        invoice_date = parse_datetime(row['InvoiceDate'])
                        if not invoice_date:
                            # Try alternative date parsing if needed
                            from datetime import datetime
                            try:
                                invoice_date = datetime.strptime(row['InvoiceDate'], '%m/%d/%Y %H:%M')
                            except ValueError:
                                print(f"⚠️ Invalid date format for invoice {invoice_no}: {row['InvoiceDate']}")

                    # Create the transaction
                    Fact_Transactions.objects.create(
                        InvoiceNo=invoice_no,
                        CustomerID=customer,
                        StockCode=product,
                        Quantity=quantity,
                        UnitPrice=unit_price,
                        TotalPrice=total_price,
                        InvoiceDate=invoice_date,
                    )
                    transactions_count += 1
                    
                    # Progress indicator for successful additions
                    if transactions_count % 1000 == 0:
                        print(f"📈 Added {transactions_count} new transactions (currently at row {row_count}, invoice {invoice_no})...")
                        
                except (ValueError, KeyError) as e:
                    print(f"⚠️ Skipping transaction row {row_count} due to data error: {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    print(f"⚠️ Unexpected error processing transaction row {row_count}: {e}")
                    error_count += 1
                    continue
            
            if not found_start:
                print(f"❌ Could not find starting invoice {start_after_invoice} in the CSV file!")
                return False
                    
        print(f"✅ Processing complete!")
        print(f"📊 Processing summary:")
        print(f"  Total rows processed after invoice {start_after_invoice}: {row_count}")
        print(f"  New transactions added: {transactions_count}")
        print(f"  Existing transactions skipped: {skipped_count}")
        print(f"  Errors encountered: {error_count}")
            
    except FileNotFoundError:
        print(f"❌ Transactions file not found: {transactions_file}")
        return False
    except Exception as e:
        print(f"❌ Error loading transactions: {e}")
        return False
    
    # Print final summary
    final_transactions = Fact_Transactions.objects.count()
    print("\n📊 Final Migration Summary:")
    print(f"Products: {Dim_Products.objects.count()}")
    print(f"Customers: {Dim_Customers.objects.count()}")
    print(f"Transactions: {final_transactions} (added {final_transactions - current_transactions})")
    
    print("🎉 Migration continuation completed successfully!")
    return True

if __name__ == "__main__":
    # Continue loading transactions from after invoice 573167 to the end
    continue_loading_transactions_from_invoice(start_after_invoice="573167")