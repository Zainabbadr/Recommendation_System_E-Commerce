from django.db import models


# 1. Dimension: Products
class Dim_Products(models.Model):
    StockCode = models.CharField(max_length=50, primary_key=True)
    Description = models.CharField(max_length=50, null=True, blank=True)
    Description_Categorize = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return self.StockCode


# 2. Dimension: Customers
class Dim_Customers(models.Model):
    CustomerID = models.IntegerField(primary_key=True)
    Country = models.CharField(max_length=100, null=True, blank=True)
    District = models.CharField(max_length=100, null=True, blank=True)
    Customer_TotalSpending = models.DecimalField(max_digits=12, decimal_places=2, null=True, blank=True)
    Segment = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return str(self.CustomerID)


# 3. Fact Table: Transactions
class Fact_Transactions(models.Model):
    InvoiceNo = models.CharField(max_length=50, primary_key=True)
    CustomerID = models.ForeignKey(Dim_Customers, on_delete=models.CASCADE)
    StockCode = models.ForeignKey(Dim_Products, on_delete=models.CASCADE)
    Quantity = models.IntegerField()
    UnitPrice = models.DecimalField(max_digits=10, decimal_places=2)
    TotalPrice = models.DecimalField(max_digits=12, decimal_places=2)
    InvoiceDate = models.DateTimeField()

    def __str__(self):
        return self.InvoiceNo
