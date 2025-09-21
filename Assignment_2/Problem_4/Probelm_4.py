
# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score


# %%
df = pd.read_csv("./find-polynomial/train.csv")
x1 = df[["x1"]].values.squeeze(axis=1)
x2 = df[["x2"]].values.squeeze(axis=1)
y = df["y"].values


# %%
df.describe()

# %%
plt.close("all")
df['x1'].plot()

# %%
df['x2'].plot()

# %%
df['y'].plot()

# %%
df.plot()

# %%
# Feature engineering
X = np.column_stack([
    x1**4,
    x2**4,
    x1**5,
    x2**5,
    x1**6,
    x2**6,
])

# %%
model = Ridge(alpha=1000)

# %%
# Define k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# %%
# Custom scoring functions
mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
r2_scorer = make_scorer(r2_score)

# %%
# Run CV
mse_scores = cross_val_score(model, X, y, cv=kf, scoring=mse_scorer)
r2_scores = cross_val_score(model, X, y, cv=kf, scoring=r2_scorer)

# %%
print("=== Ridge Regression with 5-Fold CV ===")
print("MSE per fold:", -mse_scores)
print("Mean MSE:", -mse_scores.mean())


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# %%
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# %%
model.coef_

# %%
model.intercept_

# %%
print("=== Linear Features ===")
print("MSE:", mean_squared_error(y_test, y_pred))

# %%
plt.figure(figsize=(10,5))
print(y_test.shape)
plt.scatter(np.arange(y_test.shape[0]),y_test, label="Linear", alpha=0.6, color='b')
plt.scatter(np.arange(y_test.shape[0]),y_pred, label="Linear", alpha=0.6,color='r')
plt.xlabel("test_instance #")
plt.ylabel("Predicted y")
plt.legend()
plt.title("GT vs Prediction")
plt.show()

# %%
df_test = pd.read_csv("./find-polynomial/test.csv")
df_test.head()

# %%
x1f = df_test[["x1"]].squeeze(1)
x2f = df_test[["x2"]].squeeze(1)
X_test_f = np.column_stack([
    x1f**4,
    x2f**4,
    x1f**5,
    x2f**5,
    x1f**6,
    x2f**6,
])

# %%
y_test_f = model.predict(X_test_f) 

# %%
y_test_f.shape

# %%
submission = pd.DataFrame(np.vstack([np.arange(200),y_test_f]).T,columns=["id","y"])
submission[["id"]] = submission[["id"]].astype(int)

# %%
submission

# %%
submission.to_csv('./find-polynomial/submission.csv', index=False) 


