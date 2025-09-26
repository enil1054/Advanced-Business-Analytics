import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

# ✅ 윈도우 한글 폰트 설정 (맑은 고딕)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ✅ Attrition을 숫자로 변환
df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# ✅ 수치형 변수 중 상수 컬럼 제거 (분산이 0인 컬럼 제거)
num_cols = df.select_dtypes(include=['number']).columns.drop('AttritionFlag')
num_cols = [col for col in num_cols if df[col].var() != 0]

# ✅ 수치형 변수와 이직률 상관계수
corr_with_attr = df[num_cols].corrwith(df['AttritionFlag']).sort_values(ascending=False)

print("=== 수치형 변수와 이직률 상관계수 ===")
print(corr_with_attr)

# ✅ 상위 변수 시각화
top_num_cols = corr_with_attr.abs().sort_values(ascending=False).head(8).index

plt.figure(figsize=(8, 6))
sns.heatmap(
    df[top_num_cols.tolist() + ['AttritionFlag']].corr().fillna(0).round(2),
    annot=True, cmap="RdBu_r", center=0
)
plt.title("Attrition과 주요 수치형 변수 상관관계", fontsize=14)
plt.show()

# ✅ 범주형 변수별 이직률 확인
cat_cols = df.select_dtypes(exclude=['number']).columns

print("\n=== 범주형 변수별 이직률 ===")
for col in cat_cols:
    rate = df.groupby(col, observed=True)['AttritionFlag'].mean().sort_values(ascending=False)
    print(f"\n{col} 별 이직률(%)")
    print((rate * 100).round(2))

    # 시각화 (범주가 적은 경우만)
    if df[col].nunique() <= 10:
        plt.figure(figsize=(6, 4))
        sns.barplot(x=rate.index, y=rate.values * 100, palette="Set2")
        plt.title(f"{col}별 이직률(%)")
        plt.ylabel("이직률(%)")
        plt.xticks(rotation=30)
        plt.show()
