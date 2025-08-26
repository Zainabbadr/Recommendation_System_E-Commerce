from django.db import models
import json
from django.utils import timezone

class ChatHistory(models.Model):
    chat_id = models.CharField(max_length=100, unique=True, primary_key=True)
    title = models.CharField(max_length=200, default="New Chat")
    customer_id = models.IntegerField(null=True, blank=True)
    messages_json = models.TextField(default='[]')  # Store messages as JSON
    created = models.DateTimeField(default=timezone.now)
    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-last_updated']

    def get_messages(self):
        """Get messages from JSON field"""
        try:
            return json.loads(self.messages_json)
        except (json.JSONDecodeError, TypeError):
            return []

    def set_messages(self, messages):
        """Set messages to JSON field"""
        self.messages_json = json.dumps(messages)

    def __str__(self):
        return f"Chat {self.chat_id} - {self.title}"


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
    InvoiceNo = models.CharField(max_length=50)
    CustomerID = models.ForeignKey(Dim_Customers, on_delete=models.CASCADE)
    StockCode = models.ForeignKey(Dim_Products, on_delete=models.CASCADE)
    Quantity = models.IntegerField()
    UnitPrice = models.DecimalField(max_digits=10, decimal_places=2)
    TotalPrice = models.DecimalField(max_digits=12, decimal_places=2)
    InvoiceDate = models.DateTimeField()
    TransactionID = models.AutoField(primary_key=True)

    def __str__(self):
        return self.InvoiceNo
