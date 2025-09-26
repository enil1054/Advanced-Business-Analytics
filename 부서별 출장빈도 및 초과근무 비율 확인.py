import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

# ✅ 윈도우 한글 폰트 설정 (맑은 고딕)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ✅ 부서별 출장 빈도 분포
print("\n=== 부서별 출장 빈도(%) ===")
bt_dept = pd.crosstab(df['Department'], df['BusinessTravel'], normalize='index') * 100
print(bt_dept.round(2))

plt.figure(figsize=(8,6))
bt_dept.plot(kind='bar', stacked=True, colormap="Set2", figsize=(10,6))
plt.title("부서별 출장 빈도 분포(%)", fontsize=14)
plt.ylabel("비율(%)")
plt.xticks(rotation=30)
plt.legend(title="BusinessTravel")
plt.show()


# ✅ 부서별 초과근무 비율
print("\n=== 부서별 초과근무 비율(%) ===")
ot_dept = pd.crosstab(df['Department'], df['OverTime'], normalize='index') * 100
print(ot_dept.round(2))

plt.figure(figsize=(8,6))
ot_dept.plot(kind='bar', stacked=True, colormap="Pastel1", figsize=(10,6))
plt.title("부서별 초과근무 비율(%)", fontsize=14)
plt.ylabel("비율(%)")
plt.xticks(rotation=30)
plt.legend(title="OverTime")
plt.show()
