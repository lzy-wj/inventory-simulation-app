import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

st.set_page_config(
    page_title="(s, S) 库存策略仿真平台",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class InventorySimulation:
    def __init__(self, params):
        self.params = params
        self.s = params['s']
        self.S = params['S']
        self.T = params['T']
        self.lam = params['lam']
        self.avg_demand = params.get('avg_demand', 1)
        self.L = params['L']
        self.r = params['r']
        self.K = params['K']
        self.c_unit = params['c']
        self.h = params['h']
        self.t = 0.0
        self.x = self.S
        self.y = 0
        self.C = 0.0
        self.H = 0.0
        self.R = 0.0
        self.t_C = 0.0
        self.t_O = float('inf')
        self.history = []

    def _generate_next_arrival(self):
        U = np.random.uniform(0, 1)
        inter_arrival = - (1.0 / self.lam) * np.log(U)
        return inter_arrival

    def _generate_demand_size(self):
        d = np.random.poisson(self.avg_demand)
        return max(1, d)

    def _calculate_ordering_cost(self, quantity):
        if quantity <= 0: return 0
        return self.K + self.c_unit * quantity

    def run(self):
        np.random.seed(self.params['seed'])
        self.t = 0.0
        self.x = self.S
        self.y = 0
        self.C = 0.0
        self.H = 0.0
        self.R = 0.0
        self.t_C = self._generate_next_arrival()
        self.t_O = float('inf')
        
        self.history.append({
            '时间': 0.0, '现有库存': self.x, '在途订单': self.y, 
            '事件类型': '初始化', '累计利润': 0.0, '变动量': 0
        })
        
        while True:
            next_event_time = min(self.t_C, self.t_O)
            if next_event_time > self.T:
                break
                
            if self.t_C <= self.t_O:
                event_time = self.t_C
                self.H += self.h * self.x * (event_time - self.t)
                self.t = event_time
                D = self._generate_demand_size()
                w = min(D, self.x)
                lost = D - w
                self.R += w * self.r
                self.x -= w
                triggered_order = False
                if self.x < self.s and self.y == 0:
                    self.y = self.S - self.x
                    self.t_O = self.t + self.L
                    triggered_order = True
                
                current_profit = self.R - self.C - self.H
                self.history.append({
                    '时间': self.t, '现有库存': self.x, '在途订单': self.y,
                    '事件类型': '缺货损失' if lost > 0 else ('顾客购买' if not triggered_order else '顾客购买并订货'),
                    '累计利润': current_profit, '变动量': -w
                })
                if lost > 0:
                     self.history[-1]['备注'] = f"需求:{D}, 满足:{w}, 丢失:{lost}"
                self.t_C = self.t + self._generate_next_arrival()
            else:
                event_time = self.t_O
                self.H += self.h * self.x * (event_time - self.t)
                self.t = event_time
                cost_order = self._calculate_ordering_cost(self.y)
                self.C += cost_order
                self.x += self.y
                arrived_qty = self.y
                self.y = 0
                self.t_O = float('inf')
                current_profit = self.R - self.C - self.H
                self.history.append({
                    '时间': self.t, '现有库存': self.x, '在途订单': self.y,
                    '事件类型': '订单送达', '累计利润': current_profit, '变动量': arrived_qty
                })

        self.H += self.h * self.x * (self.T - self.t)
        final_profit = self.R - self.C - self.H
        self.df_log = pd.DataFrame(self.history)
        summary = {
            'final_profit': final_profit,
            'total_revenue': self.R,
            'total_ordering_cost': self.C,
            'total_holding_cost': self.H
        }
        return self.df_log, summary

st.sidebar.header("⚙️ 实验控制台")
st.sidebar.subheader("1. 基础环境参数")
T = st.sidebar.slider("仿真周期", 10, 365, 100)
lam = st.sidebar.slider("顾客到达速率", 0.1, 10.0, 2.0)
avg_demand = st.sidebar.slider("平均单次购买量", 1, 10, 1)
L = st.sidebar.slider("订货提前期", 0.1, 10.0, 2.0)

st.sidebar.subheader("2. 财务与成本参数")
r = st.sidebar.slider("单位售价", 1.0, 200.0, 50.0, 1.0)
c = st.sidebar.slider("单位变动成本", 1.0, 200.0, 20.0, 1.0)
h = st.sidebar.slider("单位时间持有成本", 0.1, 50.0, 1.0, 0.1)
K = st.sidebar.slider("单次固定订货成本", 0.0, 500.0, 100.0, 10.0)

st.sidebar.subheader("3. 策略参数")
current_s = st.sidebar.slider("再订货点 s", 0, 200, 10)
min_S = current_s + 1
default_S = max(min_S, 40)
current_S = st.sidebar.slider("最大库存水平 S", min_S, 300, default_S)

st.sidebar.subheader("4. 实验设置")
seed = st.sidebar.number_input("随机种子", value=42, step=1)

sim_params = {
    'T': T, 'lam': lam, 'avg_demand': avg_demand, 'L': L,
    'r': r, 'c': c, 'h': h, 'K': K,
    's': current_s, 'S': current_S,
    'seed': seed
}

st.title("🏭 (s, S) 库存策略仿真与优化实验平台")
st.markdown("**运筹学与数据科学实验室 | 基于离散事件仿真**")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📖 模型原理", 
    "💻 算法源码",
    "🕹️ 单次仿真", 
    "📈 敏感性分析", 
    "🎯 策略优化"
])
# === Tab 1: 模型原理 ===
with tab1:
    st.header("带丢失销售的库存模型原理")
    
    st.subheader("📋 模型假设")
    st.markdown("""
    本模型建立在以下核心假设基础上：
    
    **基础设定：**

    • 商店仅管理**单一产品**，每单位售价为 $r$

    • 采用**连续盘点制度**，系统实时监控库存水平

    • 初始状态：$t=0$ 时库存为满仓 $S$，无在途订单

    **顾客需求过程：**

    • 顾客按速率为 $\lambda$ 的**泊松过程**到达

    • 每位顾客的需求量 $D$ 服从分布 $G$（本实验采用泊松分布）

    • 采用**丢失销售机制**：库存不足时，未满足的需求直接流失，不计入欠单

    **订货策略：**

    • 使用 $(s, S)$ 策略：当现有库存 $x < s$ 且无在途订单时，订货量 $Q = S - x$

    • **单次订货限制**：同一时刻最多只能有一个在途订单

    • **确定性提前期**：订单从发出到送达需要 $L$ 单位时间

    **成本结构：**

    • **订货成本**：$c(y) = K + c_0 \cdot y$（固定成本 $K$ + 变动成本）

    • **持有成本**：单位库存单位时间成本为 $h$，连续累积

    • **支付方式**：货到付款，订单送达时刻支付订货成本
    """)
    
    st.markdown("---")
    
    col_text, col_graph = st.columns([1, 1])
    
    with col_text:
        st.subheader("🔤 核心符号定义")
        st.markdown(r"""
        **状态变量：**
        * $t$ : 当前模拟时间
        * $x$ : 现有库存
        * $y$ : 在途订单量
        * $C$ : 累计订货成本
        * $H$ : 累计持有成本
        * $R$ : 累计收入
        
        **事件时间：**
        * $t_C$ : 下一位顾客到达的时间
        * $t_O$ : 下一批订单送达的时间
        
        **决策参数：**
        * $s$ : 再订货点
        * $S$ : 目标库存水平
        """)
        
        st.subheader("🎲 随机变量生成方法")
        st.markdown(r"""
        **顾客到达时间间隔：**
        
        泊松过程的到达间隔服从指数分布。我们使用逆变换法生成：
        
        1. 生成均匀分布随机数 $U \sim \text{Uniform}(0,1)$
        2. 利用逆变换法：$\Delta t = -\frac{1}{\lambda} \ln(U)$
        3. 下一位顾客到达时间：$t_C = t + \Delta t$
        
        **需求量生成：**
        
        单个顾客的需求量服从泊松分布：
        * 使用 NumPy 库的 `np.random.poisson()` 函数
        * 设定最小值为 1 以避免零需求
        * 泊松分布适合描述低频率的离散需求
        """)
    
    with col_graph:
        st.subheader("🔄 算法逻辑流程图")
        graph = graphviz.Digraph()
        graph.attr(rankdir='TB')
        
        graph.node('Start', '开始\n初始化 t=0, x=S', shape='oval')
        graph.node('Compare', '比较下一事件时间\nmin(t_C, t_O, T)', shape='diamond')
        
        graph.node('CustArr', '顾客到达', shape='box', style='filled', color='#e1f5fe')
        graph.node('UpdateH1', '更新持有成本 H\nt = t_C', shape='box')
        graph.node('Sell', '销售处理\nx = x - w\nR = R + r×w', shape='box')
        graph.node('CheckOrder', '库存 < s 且 y==0 ?', shape='diamond')
        graph.node('DoOrder', '触发订货\ny = S - x\nt_O = t + L', shape='box', style='filled', color='#ffe0b2')
        graph.node('NextCust', '生成下一顾客 t_C', shape='box')
        
        graph.node('OrderArr', '订单送达', shape='box', style='filled', color='#c8e6c9')
        graph.node('UpdateH2', '更新持有成本 H\nt = t_O', shape='box')
        graph.node('Pay', '支付订货成本 C\nC += c(y)', shape='box')
        graph.node('Refill', '入库\nx = x + y\ny = 0', shape='box')
        
        graph.node('End', '结束', shape='oval')
        
        graph.edge('Start', 'Compare')
        graph.edge('Compare', 'CustArr', label='t_C 最小')
        graph.edge('Compare', 'OrderArr', label='t_O 最小')
        graph.edge('Compare', 'End', label='超过 T')
        
        graph.edge('CustArr', 'UpdateH1')
        graph.edge('UpdateH1', 'Sell')
        graph.edge('Sell', 'CheckOrder')
        graph.edge('CheckOrder', 'DoOrder', label='是')
        graph.edge('CheckOrder', 'NextCust', label='否')
        graph.edge('DoOrder', 'NextCust')
        graph.edge('NextCust', 'Compare')
        
        graph.edge('OrderArr', 'UpdateH2')
        graph.edge('UpdateH2', 'Pay')
        graph.edge('Pay', 'Refill')
        graph.edge('Refill', 'Compare')

        st.graphviz_chart(graph)
    
    st.markdown("---")
    st.subheader("🎯 决策逻辑与目标")
    st.markdown(r"""
    **策略规则：**
    
    当现有库存低于再订货点且没有在途订单时，触发订货。订货量使名义库存恢复到目标水平。
    
    **目标函数：**
    
    最大化总周期内的期望利润：
    
    $$ \Pi = R - C - H $$
    
    其中：销售收入减去订货成本和持有成本
    """)

# === Tab 2: 算法源码 ===
with tab2:
    st.header("核心仿真算法实现")
    
    st.markdown("""
    本算法采用**离散事件模拟**方法，通过时间推进机制模拟系统动态。
    核心思想是：系统状态仅在特定事件发生时刻改变，在两个事件之间保持不变。
    """)
    
    st.subheader("🔧 主要技术要点")
    col_tech1, col_tech2 = st.columns(2)
    
    with col_tech1:
        st.markdown("""
        **事件驱动机制：**
        * 维护事件列表
        * 每次选择最早发生的事件
        * 更新系统时间到事件时刻
        * 执行对应的状态转移逻辑
        """)
    
    with col_tech2:
        st.markdown("""
        **状态更新策略：**
        * 持有成本累积计算
        * 库存水平实时跟踪
        * 订单触发条件检查
        * 在途订单状态管理
        """)
    
    st.subheader("📝 完整源码展示")
    st.markdown("核心仿真循环的实现逻辑，包含详细注释：")
    
    code_source = """
def run(self):
    '''主仿真循环函数'''
    
    # 初始化阶段
    np.random.seed(self.params['seed'])
    
    self.t = 0.0
    self.x = self.S
    self.y = 0
    self.C = 0.0
    self.H = 0.0
    self.R = 0.0
    
    # 生成第一个顾客到达时间
    # 使用逆变换法：从均匀分布生成指数分布
    U = np.random.uniform(0, 1)
    self.t_C = - (1.0 / self.lam) * np.log(U)
    
    self.t_O = float('inf')
    
    # 事件驱动主循环
    while True:
        next_event_time = min(self.t_C, self.t_O)
        
        if next_event_time > self.T:
            break
        
        # 情况A：顾客到达事件
        if self.t_C <= self.t_O:
            # 更新持有成本
            self.H += self.h * self.x * (self.t_C - self.t)
            
            # 推进系统时钟
            self.t = self.t_C
            
            # 生成需求量
            D = np.random.poisson(self.avg_demand)
            D = max(1, D)
            
            # 计算实际可销售数量
            w = min(D, self.x)
            lost = D - w
            
            # 更新财务状态
            self.R += w * self.r
            self.x -= w
            
            # 检查是否需要触发订货
            if self.x < self.s and self.y == 0:
                self.y = self.S - self.x
                self.t_O = self.t + self.L
            
            # 生成下一位顾客到达时间
            U = np.random.uniform(0, 1)
            inter_arrival = - (1.0 / self.lam) * np.log(U)
            self.t_C = self.t + inter_arrival
        
        # 情况B：订单送达事件
        else:
            self.H += self.h * self.x * (self.t_O - self.t)
            self.t = self.t_O
            
            # 支付订货成本
            order_cost = self.K + self.c_unit * self.y
            self.C += order_cost
            
            # 货物入库
            self.x += self.y
            
            # 重置订单状态
            self.y = 0
            self.t_O = float('inf')
    
    # 结束处理
    self.H += self.h * self.x * (self.T - self.t)
    final_profit = self.R - self.C - self.H
    
    return final_profit
"""
    st.code(code_source, language='python')
    
    st.info("💡 **实现细节**：本算法使用逆变换法从均匀分布生成指数分布的到达间隔，这是模拟泊松过程的标准方法。需求量则直接调用 NumPy 的泊松分布生成函数。")
    # === Tab 3: 单次仿真 ===
with tab3:
    st.subheader(f"当前策略: (s={current_s}, S={current_S})")
    
    sim_engine = InventorySimulation(sim_params)
    df_result, summary = sim_engine.run()
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("最终利润", f"{summary['final_profit']:,.2f}", delta_color="normal")
    kpi2.metric("总收入", f"{summary['total_revenue']:,.2f}")
    kpi3.metric("总订货成本", f"{summary['total_ordering_cost']:,.2f}", delta_color="inverse")
    kpi4.metric("总持有成本", f"{summary['total_holding_cost']:,.2f}", delta_color="inverse")

    st.markdown("### 📈 库存状态随时间变化图")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    times = df_result['时间']
    inventory = df_result['现有库存']
    
    color_inv = 'tab:blue'
    ax1.set_xlabel('仿真时间')
    ax1.set_ylabel('现有库存量', color=color_inv, fontsize=12)
    ax1.step(times, inventory, where='post', color=color_inv, label='现有库存', alpha=0.8, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_inv)
    
    ax1.axhline(y=current_s, color='orange', linestyle='--', label='再订货点', alpha=0.8)
    ax1.axhline(y=current_S, color='green', linestyle='--', label='最大库存', alpha=0.8)
    ax1.fill_between(times, 0, inventory, step='post', color=color_inv, alpha=0.1)

    orders_placed = df_result[df_result['事件类型'] == '顾客购买并订货']
    orders_arrived = df_result[df_result['事件类型'] == '订单送达']
    stockouts = df_result[df_result['事件类型'] == '缺货损失']

    if not orders_placed.empty:
        ax1.scatter(orders_placed['时间'], orders_placed['现有库存'], 
                   color='orange', marker='o', s=80, zorder=5, label='触发订货点')
    
    if not orders_arrived.empty:
        ax1.scatter(orders_arrived['时间'], orders_arrived['现有库存'], 
                   color='green', marker='^', s=100, zorder=5, label='订单送达')
                   
    if not stockouts.empty:
        ax1.scatter(stockouts['时间'], stockouts['现有库存'], 
                   color='red', marker='x', s=100, zorder=5, label='发生缺货')

    ax2 = ax1.twinx()
    color_profit = 'tab:gray'
    ax2.set_ylabel('累计利润', color=color_profit, fontsize=12)
    ax2.plot(times, df_result['累计利润'], color=color_profit, linestyle=':', linewidth=1.5, label='累计利润曲线')
    ax2.tick_params(axis='y', labelcolor=color_profit)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', frameon=True, fancybox=True)
    
    plt.title(f"仿真轨迹: T={T}, L={L}, s={current_s}, S={current_S}")
    st.pyplot(fig)

    col_pie, col_data = st.columns([1, 2])
    
    with col_pie:
        st.markdown("#### 成本结构分布")
        cost_values = [summary['total_holding_cost'], summary['total_ordering_cost']]
        cost_labels = ['总持有成本', '总订货成本']
        
        if sum(cost_values) > 0:
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(cost_values, labels=cost_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
            ax_pie.set_title("运营成本构成")
            st.pyplot(fig_pie)
        else:
            st.info("暂无成本产生")
        
    with col_data:
        st.markdown("#### 事件日志明细")
        st.dataframe(
            df_result[['时间', '事件类型', '现有库存', '变动量', '累计利润']].style.format({
                '时间': "{:.2f}",
                '累计利润': "{:.2f}",
                '现有库存': "{:.0f}"
            }),
            height=300
        )

# === Tab 4: 敏感性分析 ===
with tab4:
    st.header("📈 单参数敏感性分析")
    st.markdown("分析当改变某一个参数时，对最终利润的影响趋势。")
    
    col_param, col_range = st.columns([1, 2])
    
    with col_param:
        analysis_target = st.selectbox("选择分析变量", ["再订货点 s", "最大库存 S", "订货提前期 L"])
    
    results_sensitivity = []
    x_vals = []
    
    if analysis_target == "再订货点 s":
        st.caption(f"固定 S={current_S}, 变化 s")
        range_values = range(0, current_S) 
        x_label = "再订货点 s"
        for val in range_values:
            p = sim_params.copy()
            p['s'] = val
            _, s = InventorySimulation(p).run()
            results_sensitivity.append(s['final_profit'])
            x_vals.append(val)
            
    elif analysis_target == "最大库存 S":
        st.caption(f"固定 s={current_s}, 变化 S")
        range_values = range(current_s + 1, current_s + 51)
        x_label = "最大库存 S"
        for val in range_values:
            p = sim_params.copy()
            p['S'] = val
            _, s = InventorySimulation(p).run()
            results_sensitivity.append(s['final_profit'])
            x_vals.append(val)
            
    elif analysis_target == "订货提前期 L":
        st.caption(f"固定 s, S, 变化 L")
        range_values = np.linspace(0.5, 10.0, 20)
        x_label = "订货提前期 L"
        for val in range_values:
            p = sim_params.copy()
            p['L'] = val
            _, s = InventorySimulation(p).run()
            results_sensitivity.append(s['final_profit'])
            x_vals.append(val)

    fig_sens, ax_sens = plt.subplots(figsize=(10, 4))
    ax_sens.plot(x_vals, results_sensitivity, marker='o', linestyle='-', color='purple')
    ax_sens.set_xlabel(x_label)
    ax_sens.set_ylabel("总利润")
    ax_sens.set_title(f"敏感性分析: {x_label} 对利润的影响")
    ax_sens.grid(True, linestyle='--', alpha=0.6)
    
    max_y = max(results_sensitivity)
    max_x = x_vals[results_sensitivity.index(max_y)]
    ax_sens.annotate(f'峰值: {max_y:.0f}', xy=(max_x, max_y), xytext=(max_x, max_y*1.05),
                     arrowprops=dict(facecolor='black', shrink=0.05))
    
    st.pyplot(fig_sens)

# === Tab 5: 策略优化 ===
with tab5:
    st.header("🎯 全局策略优化")
    st.markdown("遍历不同的组合，寻找利润最大化的参数配置。")
    
    col_opt_1, col_opt_2 = st.columns(2)
    with col_opt_1:
        s_max_search = st.slider("s 搜索上限", 10, 50, 20)
    with col_opt_2:
        S_max_search = st.slider("S 搜索上限", 10, 100, 60)

    if st.button("🚀 开始优化计算"):
        progress_bar = st.progress(0)
        
        heatmap_data = []
        best_profit = -float('inf')
        best_config = (0, 0)
        
        s_range = range(0, s_max_search + 1, 2)
        total_steps = len(s_range)
        
        for i, s_val in enumerate(s_range):
            S_range = range(s_val + 5, S_max_search + 1, 5) 
            for S_val in S_range:
                p = sim_params.copy()
                p['s'] = s_val
                p['S'] = S_val
                _, res = InventorySimulation(p).run()
                
                profit = res['final_profit']
                heatmap_data.append({'s': s_val, 'S': S_val, 'Profit': profit})
                
                if profit > best_profit:
                    best_profit = profit
                    best_config = (s_val, S_val)
            
            progress_bar.progress((i + 1) / total_steps)
        
        df_heatmap = pd.DataFrame(heatmap_data)
        
        st.success(f"✅ 优化完成! 建议策略: s* = {best_config[0]}, S* = {best_config[1]}, 预期利润: {best_profit:,.2f}")
        
        pivot_table = df_heatmap.pivot(index='s', columns='S', values='Profit')
        
        fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=False, fmt=".0f", cmap="viridis", ax=ax_hm, cbar_kws={'label': '总利润'})
        ax_hm.set_title("利润热力图")
        ax_hm.invert_yaxis()
        st.pyplot(fig_hm)