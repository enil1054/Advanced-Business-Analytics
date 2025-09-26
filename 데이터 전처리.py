import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import rc

# ✅ 윈도우 한글 폰트 설정 (맑은 고딕)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 데이터 크기 확인
print("데이터 크기(행, 열):", df.shape)

# 데이터 타입 및 기본 정보
print("\n데이터 타입 및 결측치 확인:")
print(df.info())

# 결측치 확인
print("\n결측치 개수:")
print(df.isnull().sum())

# 기초 통계량
print("\n수치형 변수 기초 통계량:")
print(df.describe())

print("\n범주형 변수 기초 통계량:")
print(df.describe(include=['object']))

# 이상치 탐색 (IQR 방식)
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

numeric_cols = df.select_dtypes(include=[np.number]).columns

print("\n이상치 탐색 결과:")
for col in numeric_cols:
    outliers = detect_outliers_iqr(df, col)
    print(f"{col}: 이상치 {len(outliers)}개")

# 시각적으로 이상치 확인 (Boxplot)
for col in numeric_cols:
    plt.figure(figsize=(6,4))
    plt.boxplot(df[col].dropna())
    plt.title(f"{col} - 이상치 탐색(Boxplot)")
    plt.ylabel(col)
    plt.show()


# 불만족(1~2), 만족(3~4) 그룹화
df['SatisfactionGroup'] = df['EnvironmentSatisfaction'].apply(
    lambda x: '불만족(1~2)' if x <= 2 else '만족(3~4)'
)

# 부서별 인원 수 집계
dept_satisfaction = pd.crosstab(df['Department'], df['SatisfactionGroup'])

dept_satisfaction_ratio = pd.crosstab(
    df['Department'],
    df['SatisfactionGroup'],
    normalize='index'
).round(2)

print("부서별 불만족/만족 인원수:")
print(dept_satisfaction)

print("\n부서별 불만족/만족 비율:")
print(dept_satisfaction_ratio)


dept_satisfaction_ratio.plot(
    kind='bar', stacked=True, figsize=(8,5), colormap='RdYlGn'
)
plt.title("부서별 환경 만족도 (비율)")
plt.ylabel("비율")
plt.legend(title="만족도 그룹")
plt.show()
