path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
print("✅ Dataset downloaded at:", path)
csv_path = os.path.join(path, "heart.csv")
df = pd.read_csv(csv_path)

print("\n=== Dataset Preview ===")
print(df.head())
print("\nColumns:\n", df.columns)
print("\nMissing values:\n", df.isnull().sum())

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['HeartDisease']).values
y = df['HeartDisease'].values

from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)
