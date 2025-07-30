import os
import sys
import django

sys.path.append("C:/Users/NewTech/Desktop/New folder/worldtour")  # أو استخدمي os.path

 
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'worldtour.settings')

 
django.setup()

from europe.models import Dim_Products, Dim_Customers, Fact_Transactions

import csv
from django.utils.dateparse import parse_datetime

# 1. Load Dim_Products
with open('C:/Users/NewTech/Desktop/New folder/worldtour/europe/Dim_Products.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        Dim_Products.objects.update_or_create(
            StockCode=row['StockCode'],
            defaults={
                'Description': row.get('Description') or None,
                'Description_Categorize': row.get('Description_Categorize') or None,
            }
        )

# 2. Load Dim_Customers
with open('C:/Users/NewTech/Desktop/New folder/worldtour/europe/Dim_Customers.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        Dim_Customers.objects.update_or_create(
            CustomerID=int(float(row['CustomerID'])),
            defaults={
                'Country': row.get('Country') or None,
                'District': row.get('District') or None,
                'Customer_TotalSpending': row.get('Customer_TotalSpending') or 0,
                'Segment': row.get('Segment') or None,
            }
        )

# 3. Load Fact_Transactions
with open('C:/Users/NewTech/Desktop/New folder/worldtour/europe/Fact_Transactions.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        customer = Dim_Customers.objects.get(CustomerID=int(float(row['CustomerID'])))
        product = Dim_Products.objects.get(StockCode=row['StockCode'])

        Fact_Transactions.objects.update_or_create(
            InvoiceNo=row['InvoiceNo'],
            defaults={
                'CustomerID': customer,
                'StockCode': product,
                'Quantity': int(row['Quantity']),
                'UnitPrice': float(row['UnitPrice']),
                'TotalPrice': float(row['TotalPrice']),
                'InvoiceDate': parse_datetime(row['InvoiceDate']),
            }
        )
