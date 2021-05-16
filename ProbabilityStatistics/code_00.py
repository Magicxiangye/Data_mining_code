# 试一下常用的数据科学的库
# numpy是常用的库
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
import matplotlib.pyplot as plt
# 数据可视化的库
import seaborn as sns
'''
Scipy是一个高级的科学计算库，Scipy一般都是操控Numpy数组来进行科学计算，
Scipy包含的功能有最优化、线性代数、积分、插值、拟合、特殊函数、快速傅里叶变换、
信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。
'''

# 统计分布的可视化
def Binomial_distribution():
    # 二项分布
    # B(10, 0.5)---B（n, p）
    # 二项分布成功的次数（X轴）plot画图使用
    x = range(11)
    # stats的二项随机抽样,还是B(n, p)抽三百次的随机数
    t = stats.binom.rvs(10, 0.5, size=1000)
    # 得出的概率（离散型随机变量）
    p = stats.binom.pmf(x, 10, 0.5)

    # 可视化
    # 设置图片的行列数，类似二维数组
    fig, ax = plt.subplots(1, 1)
    # seaborn数据的可视化
    # 先来一个直方图
    # 通过hist和kde参数调节
    # 是否显示直方图及核密度估计(默认hist,kde均为True)
    sns.distplot(t, bins=10, hist_kws={'density': True}, kde=False, label='300 sample')
    # 接下来散点图
    sns.scatterplot(x, p, color='purple')
    # 折线图
    sns.lineplot(x, p, color='purple', label='True mass density')
    plt.title("binomial_distribution")
    # 加图例
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.show()

# 泊松分布
# 比较不同参数λ对应的概率质量函数，可以验证随着参数增大，
# 泊松分布开始逐渐变得对称，分布也越来越均匀，趋近于正态分布
def poission_distribution():
    # 设定的次数
    x = range(11)
    # 设置λ为2
    t = stats.poisson.rvs(2, size=500)
    p = stats.poisson.pmf(x, 2)

    # 可视化
    fig, ax = plt.subplots(1, 1)
    sns.distplot(x, bins=10, hist_kws={'density: True'}, kde=False, label='distplot 500')
    # 散点图
    sns.scatterplot(x, p, color='purple')
    # 折线图
    sns.lineplot(x, p, color='purple', label='mass density')
    plt.title('possion distribution')
    plt.show()


# 假设检验的方法
# 正态检验
# Shapiro-Wilk Test是一种经典的正态检验方法。
def normal_judge():
    # 先创建使用检验的两个数据
    data_nonnormal = np.random.exponential(size=1000)
    data_normal = np.random.normal(size=1000)

    # 使用检验的方式
    def test(data):
        # 置信水平和单侧的临界值
        stat, p = stats.shapiro(data)
        if p > 0.05:
            return 'stat={:.3f}, p = {:.3f}, probably gaussian'.format(stat, p)
        else:
            return 'stat={:.3f}, p = {:.3f}, probably not gaussian'.format(stat, p)

    # 使用
    test(data_nonnormal)
    test(data_normal)





if __name__ == "__main__":
    Binomial_distribution()