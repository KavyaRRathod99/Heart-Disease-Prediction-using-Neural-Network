from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

probs = model.predict(Xte).ravel()
preds = (probs > 0.5).astype(int)
test_acc = accuracy_score(yte, preds)

print(f"\n✅ FINAL TEST ACCURACY: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(yte, preds, digits=4))

# Confusion Matrix
cm = confusion_matrix(yte, preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix — Heart Disease Detection')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.tight_layout()
plt.show()
