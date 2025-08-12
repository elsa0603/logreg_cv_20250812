import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, io, json, h5py, joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

st.set_page_config(page_title="Logistic Regression CV + GridSearch", layout="wide")

st.title("🔎 Logistic Regression + 交叉驗證 + GridSearchCV")

# -----------------------------
# Sidebar: data source
# -----------------------------
st.sidebar.header("📂 資料來源")
data_source = st.sidebar.radio("選擇資料來源", ["Iris（二元）", "上傳 CSV"], index=0)

def load_iris_binary():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df[df["target"] < 2].reset_index(drop=True)  # setosa vs versicolor
    df["target"] = df["target"].astype(int)
    return df, "target"

df = None
target_col = None

if data_source == "Iris（二元）":
    df, target_col = load_iris_binary()
    st.subheader("Iris（僅 setosa 與 versicolor）")
    st.dataframe(df.head())
else:
    file = st.sidebar.file_uploader("上傳 CSV 檔", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.subheader("已上傳資料（前 5 筆）")
        st.dataframe(df.head())
        # select target column
        with st.sidebar:
            target_col = st.selectbox("選擇目標欄 (target)", options=df.columns, index=len(df.columns)-1)
    else:
        st.info("請於側欄上傳 CSV。")
        st.stop()

# -----------------------------
# Sidebar: feature selection
# -----------------------------
st.sidebar.header("🔧 特徵與預處理")
if df is not None and target_col is not None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_candidates = [c for c in numeric_cols if c != target_col]
    if len(feature_candidates) == 0:
        st.error("找不到可用的數值特徵欄。請確認資料或轉換欄位型態。")
        st.stop()

    selected_features = st.sidebar.multiselect(
        "選擇特徵欄（至少 1 個，若選 2 個可視覺化決策邊界）",
        options=feature_candidates,
        default=feature_candidates[:2] if len(feature_candidates)>=2 else feature_candidates
    )

    if len(selected_features) == 0:
        st.warning("請至少選擇 1 個特徵欄。")
        st.stop()

    use_standardize = st.sidebar.checkbox("使用 StandardScaler 標準化", value=True)

# -----------------------------
# Sidebar: train/test split & CV
# -----------------------------
st.sidebar.header("🧪 切分與交叉驗證設定")
test_size = st.sidebar.slider("測試集比例", 0.1, 0.5, 0.2, 0.05)
cv_folds = st.sidebar.slider("交叉驗證折數 (K)", 3, 10, 5, 1)
random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

# -----------------------------
# Sidebar: Scoring
# -----------------------------
st.sidebar.header("🏁 評分指標")
scoring = st.sidebar.selectbox(
    "scoring",
    options=["accuracy", "f1", "roc_auc"],
    index=0,
    help="若資料不平衡，建議嘗試 f1 或 roc_auc（僅二元適用）"
)

# -----------------------------
# Sidebar: Grid parameters
# -----------------------------
st.sidebar.header("🧭 超參數搜尋（GridSearch）")
use_grid = st.sidebar.checkbox("啟用 GridSearchCV", value=True)

# C range
st.sidebar.markdown("**C 值範圍（對數刻度）**")
c_min_exp, c_max_exp = st.sidebar.slider("log10(C) 範圍", -3, 3, (-3, 3))
c_points = st.sidebar.slider("C 取樣點數", 3, 15, 7)
C_values = np.logspace(c_min_exp, c_max_exp, c_points)

solvers = st.sidebar.multiselect("solver", ["liblinear", "saga"], default=["liblinear", "saga"])
penalties = st.sidebar.multiselect("penalty", ["l1", "l2", "elasticnet"], default=["l1", "l2", "elasticnet"])
l1_ratios = st.sidebar.multiselect("l1_ratio (僅 elasticnet)", [0.1, 0.5, 0.9], default=[0.5])
class_weight_opt = st.sidebar.selectbox("class_weight", [None, "balanced"], index=0)

# Threshold
st.sidebar.header("📈 決策閾值（預測機率>閾值判為正類）")
threshold = st.sidebar.slider("threshold", 0.1, 0.9, 0.5, 0.05)

# -----------------------------
# Prepare data
# -----------------------------
X = df[selected_features].values
y = df[target_col].values

# ensure binary labels for f1/roc_auc
unique_y = np.unique(y)
if scoring in ("f1", "roc_auc") and len(unique_y) != 2:
    st.warning("選擇了二元評分指標，但目標欄不是二元。將改用 accuracy。")
    scoring = "accuracy"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, stratify=y if len(unique_y)>1 else None, random_state=random_state
)

# -----------------------------
# Build pipeline
# -----------------------------
steps = []
if use_standardize:
    steps.append(("scaler", StandardScaler()))
steps.append(("clf", LogisticRegression(max_iter=5000)))
pipe = Pipeline(steps)

# Parameter grid
param_grid = []
if use_grid:
    # Two dicts to respect solver/penalty compatibility
    if "liblinear" in solvers:
        param_grid.append({
            "clf__solver": ["liblinear"],
            "clf__penalty": [p for p in penalties if p in ("l1", "l2")],
            "clf__C": C_values,
            "clf__class_weight": [class_weight_opt]
        })
    if "saga" in solvers:
        grid_saga = {
            "clf__solver": ["saga"],
            "clf__penalty": [p for p in penalties if p in ("l1", "l2", "elasticnet")],
            "clf__C": C_values,
            "clf__class_weight": [class_weight_opt]
        }
        if "elasticnet" in grid_saga["clf__penalty"]:
            grid_saga["clf__l1_ratio"] = l1_ratios if len(l1_ratios) > 0 else [0.5]
        param_grid.append(grid_saga)

# -----------------------------
# Train
# -----------------------------
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

if use_grid and len(param_grid) > 0:
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True
    )
    grid.fit(X_train, y_train)
    best_est = grid.best_estimator_
    best_params = grid.best_params_
    best_cv_mean = grid.best_score_
    st.success(f"最佳CV分數 ({scoring}): {best_cv_mean:.4f}")
    st.write("最佳參數組合：", best_params)

    # Top-N results
    with st.expander("🔎 查看前 5 名參數組合"):
        means = grid.cv_results_["mean_test_score"]
        stds = grid.cv_results_["std_test_score"]
        params = grid.cv_results_["params"]
        top = sorted(zip(means, stds, params), key=lambda x: x[0], reverse=True)[:5]
        top_df = pd.DataFrame(
            [dict(score_mean=m, score_std=s, **p) for m, s, p in top]
        )
        st.dataframe(top_df)
else:
    # No grid search: just fit the pipeline
    best_est = pipe.fit(X_train, y_train)
    st.info("未啟用 GridSearchCV，已直接訓練基本模型。")
    best_params = None

# -----------------------------
# Evaluate
# -----------------------------
st.header("📊 模型評估（測試集）")


y_prob = None
if len(np.unique(y_train)) == 2:
    # Binary: allow custom threshold
    y_prob = best_est.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > threshold).astype(int)
else:
    y_pred = best_est.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"**Accuracy:** {acc:.4f}")

if len(np.unique(y_train)) == 2 and y_prob is not None:
    try:
        auc = roc_auc_score(y_test, y_prob)
        st.write(f"**ROC AUC (for reference):** {auc:.4f}")
    except Exception:
        pass

st.write("**混淆矩陣**")
st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))

st.write("**分類報告**")
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).T)

# -----------------------------
# Visualization (only when exactly 2 numeric features selected)
# -----------------------------
if len(selected_features) == 2:
    st.header("🗺️ 決策邊界（2 特徵）")

    # In this app, X_train/X_test already correspond to selected_features in order
    X2_train = X_train[:, [0, 1]]
    X2_test = X_test[:, [0, 1]]

    # Build model using the learned LR settings
    steps2 = []
    if use_standardize:
        steps2.append(("scaler", StandardScaler()))

    # Extract LR config from fitted pipeline
    lr_step = None
    for name, step in best_est.steps:
        if isinstance(step, LogisticRegression):
            lr_step = step
            break
    if lr_step is None:
        lr_step = LogisticRegression(max_iter=5000)

    steps2.append(("clf", LogisticRegression(
        max_iter=lr_step.max_iter,
        solver=lr_step.solver,
        penalty=lr_step.penalty,
        C=lr_step.C,
        l1_ratio=getattr(lr_step, "l1_ratio", None),
        class_weight=lr_step.class_weight
    )))
    pipe2 = Pipeline(steps2)
    pipe2.fit(X2_train, y_train)

    x_min, x_max = X2_train[:, 0].min() - 1.0, X2_train[:, 0].max() + 1.0
    y_min, y_max = X2_train[:, 1].min() - 1.0, X2_train[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    if len(np.unique(y_train)) == 2:
        Z = pipe2.predict_proba(grid_points)[:, 1].reshape(xx.shape)
    else:
        Z = pipe2.predict(grid_points).reshape(xx.shape)

    fig = plt.figure(figsize=(6, 5))
    cs = plt.contourf(xx, yy, Z, levels=20, alpha=0.6)
    if len(np.unique(y_train)) == 2:
        plt.contour(xx, yy, Z, levels=[0.5])
    plt.scatter(X2_train[:, 0], X2_train[:, 1], marker='o', label='train')
    plt.scatter(X2_test[:, 0], X2_test[:, 1], marker='x', label='test')
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title("Decision Boundary (Logistic Regression)")
    plt.legend()
    st.pyplot(fig)
else:
    st.info("若剛好選擇 **2 個數值特徵**，將顯示決策邊界圖。")

# -----------------------------
# Downloads
# -----------------------------
st.header("💾 下載模型")


col1, col2, col3 = st.columns(3)

with col1:
    pkl_bytes = pickle.dumps(best_est)
    st.download_button(
        label="下載 .pkl",
        data=pkl_bytes,
        file_name="model.pkl",
        mime="application/octet-stream"
    )

with col2:
    buf = io.BytesIO()
    joblib.dump(best_est, buf)
    buf.seek(0)
    st.download_button(
        label="下載 .joblib",
        data=buf.getvalue(),
        file_name="model.joblib",
        mime="application/octet-stream"
    )

with col3:
    # NOTE: For sklearn, .h5 is not native. We export a lightweight HDF5 containing parameters.
    # This is mainly for interoperability or inspection, not a drop-in model file.
    h5buf = io.BytesIO()
    with h5py.File(h5buf, "w") as h5f:
        grp = h5f.create_group("model_info")
        grp.attrs["framework"] = "scikit-learn"
        grp.attrs["estimator"] = str(best_est)
        grp.create_dataset("selected_features", data=np.array(selected_features, dtype="S"))
        # Try to extract LR step to save coefficients/intercept/classes
        try:
            lr = None
            for name, step in best_est.steps:
                if isinstance(step, LogisticRegression):
                    lr = step
                    break
            if lr is not None:
                lr_grp = h5f.create_group("logistic_regression")
                lr_grp.create_dataset("coef_", data=lr.coef_)
                lr_grp.create_dataset("intercept_", data=lr.intercept_)
                lr_grp.create_dataset("classes_", data=lr.classes_)
                # store config
                cfg = dict(
                    solver=lr.solver,
                    penalty=lr.penalty,
                    C=float(lr.C),
                    max_iter=int(lr.max_iter),
                    class_weight=("balanced" if lr.class_weight == "balanced" else None),
                    l1_ratio=(None if getattr(lr, "l1_ratio", None) is None else float(lr.l1_ratio))
                )
                lr_grp.attrs["config_json"] = json.dumps(cfg)
        except Exception as e:
            grp.attrs["warning"] = f"Could not extract LR parameters: {e}"
    h5buf.seek(0)
    st.download_button(
        label="下載 .h5（參數導出）",
        data=h5buf.getvalue(),
        file_name="model_params.h5",
        mime="application/octet-stream",
        help="此 .h5 檔僅儲存係數/結構資訊，非可直接 `.predict()` 的 sklearn 物件。"
    )
