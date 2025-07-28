from preprocessing import preprocess_all

df, rfm = preprocess_all()

# عرض بعض النتائج للتأكيد
print("✅ DataFrame (df) preview:")
print(df.head())

print("\n✅ RFM Segmentation preview:")
print(rfm.head())
