import pandas as pd
import numpy as np

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("\n=== [0] 데이터 개요 ===")
print("크기(행, 열):", df.shape)

const_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
if const_cols:
    print("상수 컬럼(참고):", const_cols)

# Attrition 이진 플래그 추가
if df['Attrition'].dtype == 'O':
    df['AttritionFlag'] = df['Attrition'].map({'Yes': 1, 'No': 0})
else:
    df['AttritionFlag'] = df['Attrition'].astype(int)

# 편의용 연속형/범주형 분리
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\n=== [1] 기본 특성 ===")
print("\n나이(Age) 요약:")
print(df['Age'].describe())

print("\n성별(Gender) 분포:")
print(df['Gender'].value_counts(dropna=False))

print("\n학력(Education) 분포:")
print(df['Education'].value_counts(dropna=False))

print("\n학문 분야(EducationField) 분포:")
print(df['EducationField'].value_counts(dropna=False))

print("\n부서(Department) 인원수:")
print(df['Department'].value_counts(dropna=False))

print("\n직무(JobRole) 인원수:")
print(df['JobRole'].value_counts(dropna=False))

print("\n직급(JobLevel) 인원수:")
print(df['JobLevel'].value_counts(dropna=False))

print("\n출장(BusinessTravel) 분포:")
print(df['BusinessTravel'].value_counts(dropna=False))

print("\n초과근무(OverTime) 분포:")
print(df['OverTime'].value_counts(dropna=False))

print("\n=== [2] 보상/급여 ===")
print("\n월급(MonthlyIncome) 요약:")
print(df['MonthlyIncome'].describe())

print("\n시급(HourlyRate) 요약:")
print(df['HourlyRate'].describe())

print("\n일급(DailyRate) 요약:")
print(df['DailyRate'].describe())

if 'PercentSalaryHike' in df.columns:
    print("\n급여 인상률(PercentSalaryHike) 요약:")
    print(df['PercentSalaryHike'].describe())

print("\n=== [3] 근속/경력 ===")
print("\n근속연수(YearsAtCompany) 요약:")
print(df['YearsAtCompany'].describe())

if 'TotalWorkingYears' in df.columns:
    print("\n총 근무연수(TotalWorkingYears) 요약:")
    print(df['TotalWorkingYears'].describe())

for col in ['YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
    if col in df.columns:
        print(f"\n{col} 요약:")
        print(df[col].describe())

print("\n=== [4] 만족도 ===")
for col in ['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']:
    if col in df.columns:
        print(f"\n{col} 분포:")
        print(df[col].value_counts(dropna=False))

print("\n=== [5] 성과 ===")
if 'PerformanceRating' in df.columns:
    print("\n성과평가(PerformanceRating) 분포:")
    print(df['PerformanceRating'].value_counts(dropna=False))

print("\n=== [6] 이직(Attrition) ===")
print("\n전체 이직률:", round(df['AttritionFlag'].mean()*100, 2), "%")

# 부서/출장/초과근무/직급/직무/연령대별 이직률
def rate_by(col):
    s = df.groupby(col, observed=True)['AttritionFlag'].mean().sort_values(ascending=False)
    return (s*100).round(2)

print("\n부서별 이직률(%)")
print(rate_by('Department'))

print("\n출장 형태별 이직률(%)")
print(rate_by('BusinessTravel'))

print("\n초과근무별 이직률(%)")
print(rate_by('OverTime'))

print("\n직급(JobLevel)별 이직률(%)")
print(rate_by('JobLevel'))

print("\n직무(JobRole)별 이직률(%)")
print(rate_by('JobRole'))

# 연령대 구간별 이직률
df['Age_bin'] = pd.cut(df['Age'], bins=[18,25,35,45,55,65], labels=['18-25','26-35','36-45','46-55','56-65'], right=True, include_lowest=True)
print("\n연령대별 이직률(%)")
print(rate_by('Age_bin'))

# DailyRate 구간별 이직률(사분위)
dr_q = df['DailyRate'].quantile([0, .25, .5, .75, 1.0]).tolist()
df['DailyRate_bin'] = pd.cut(df['DailyRate'], bins=dr_q, include_lowest=True)
print("\nDailyRate 사분위 구간별 이직률(%)")
print(rate_by('DailyRate_bin'))

print("\n=== [7] 빠른 품질 점검 ===")
print("\n결측치 개수(상위 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

print("\n수치형 상관(상위 5x5 미리보기):")
print(df[num_cols].corr(numeric_only=True).round(2).head())
