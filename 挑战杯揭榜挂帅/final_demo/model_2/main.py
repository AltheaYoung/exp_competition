import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import random

# 从您另外两个文件中导入定义
from final_demo.INTERFACE_FINAL.dataclass import Flight, Airport
from final_demo.INTERFACE_FINAL.datapre import load_data_from_csv

# ==============================================================================
# 模块1: 加载真实的 "出港" 流量预测数据
# ==============================================================================
def load_predicted_departure_capacity(capacity_filepath: str) -> Dict[datetime, int]:
    """
    (新) 从CSV文件加载特定机场的 "出港" 预测流量（容量）。

    Args:
        capacity_filepath (str): 流量预测CSV文件的路径。

    Returns:
        Dict[datetime, int]: 字典，键是小时开始的时间戳，值是该小时的出港容量。
    """
    capacity_forecast = {}
    print(f"\n开始从 {capacity_filepath} 加载出港流量预测数据...")
    try:
        df = pd.read_csv(capacity_filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        for _, row in df.iterrows():
            # 直接将预测流量作为出港容量
            capacity_forecast[row['timestamp']] = int(row['actual_flow'])+1
            
        print(f"成功加载 {len(capacity_forecast)} 个小时的出港容量数据。")
        return capacity_forecast
        
    except Exception as e:
        print(f"读取容量预测文件时发生错误: {e}")
        return {}

def query_predicted_capacity(forecast: Dict, time: datetime, default_capacity: int = 0) -> int:
    """从加载的预测中查询特定时间的出港容量。"""
    query_time = time.replace(minute=0, second=0, microsecond=0)
    return forecast.get(query_time, default_capacity)

# ==============================================================================
# 模块2: 新增 - 仿真后分析函数
def analyze_congestion_periods(hourly_log: pd.DataFrame, threshold: int = 10) -> Dict:
    """
    (正确版本) 分析小时日志，找出并报告所有独立的积压时段。
    """
    log_df = hourly_log.copy()
    log_df['is_congested'] = log_df['departure_queue_size'] > threshold
    
    if not log_df['is_congested'].any():
        return {"summary": "当日未监测到积压时段", "congestion_periods": []}

    log_df['block_id'] = (log_df['is_congested'] != log_df['is_congested'].shift()).cumsum()
    congested_blocks = log_df[log_df['is_congested']]

    all_congestion_periods = []
    for _, period_df in congested_blocks.groupby('block_id'):
        start_hour = period_df['timestamp'].min()
        end_hour = period_df['timestamp'].max()
        duration_hours = len(period_df)
        
        # 注意: 列名应与 hourly_congestion_log 中创建的一致
        peak_row = period_df.loc[period_df['departure_queue_size'].idxmax()]
        peak_hour = peak_row['timestamp']
        peak_value = peak_row['departure_queue_size']
        
        all_congestion_periods.append({
            "start_hour": start_hour, "end_hour": end_hour, "duration_hours": int(duration_hours),
            "peak_hour": peak_hour, "peak_value": peak_value
        })

    return {
        "summary": f"当日共监测到 {len(all_congestion_periods)} 个积压时段",
        "congestion_periods": all_congestion_periods
    }

# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":
    
    # --- 步骤零：定义目标和加载数据 ---
    TARGET_AIRPORT = "ZGGG"  # 聚焦机场：广州白云
    CONGESTION_THRESHOLD = 10 # 积压阈值：10架飞机
    
    # 加载航班计划数据（现在只包含ZGGG的出港航班）
    flights_filepath = r'C:\Users\Administrator\Desktop\Q2\project-out\zggg_departures_only_2025-05-23.csv'
    flight_objects, airport_objects = load_data_from_csv(flights_filepath)

    if not flight_objects:
        print("没有加载到有效的航班数据，程序退出。")
        exit()

    # 加载广州机场的真实 "出港" 流量预测
    capacity_filepath = r'C:\Users\Administrator\Desktop\Q2\project-out\prediction_for_may_20_23_final.csv' # 确保这是你的流量预测文件
    zggg_capacity_forecast = load_predicted_departure_capacity(capacity_filepath)

    # --- 步骤一：仿真初始化 ---
    sim_start_time = datetime(2025, 5, 23, 0, 0)
    sim_end_time = sim_start_time + timedelta(hours=24)
    time_step = timedelta(minutes=1)
    
    current_time = sim_start_time
    hourly_congestion_log = [] # 新增：用于记录每小时的积压情况
    delayed_flights_log = []
    airport_capa_log=[]
    print(f"\n--- ZGGG出港仿真开始 --- \n时间范围: {sim_start_time} to {sim_end_time}\n")

    # --- 步骤二：主仿真循环 ---
    while current_time < sim_end_time:
        
        # 2.1 更新ZGGG的出港容量
        zggg_airport = airport_objects.get(TARGET_AIRPORT)
        if zggg_airport:
            capacity_now = query_predicted_capacity(zggg_capacity_forecast, current_time, zggg_airport.standard_departure_capacity)
            zggg_airport.update_capacity(capacity_now, 0) # 进港容量设为0，我们不关心

        # 2.2 处理航班状态转换 (只关心出港)
        for flight in flight_objects.values():
            if flight.status == "Scheduled" and flight.departure_airport == TARGET_AIRPORT and flight.scheduled_departure_time == current_time:
                flight.status = "Awaiting Takeoff"
                zggg_airport.departure_queue.append(flight.flight_id)
            
        # 2.3 处理ZGGG的出港队列
        if zggg_airport:
            zggg_airport.departure_slot_accumulator += zggg_airport.current_departure_capacity / 60.0
            num_to_takeoff = int(zggg_airport.departure_slot_accumulator)
            if num_to_takeoff > 0 and zggg_airport.departure_queue:
                for _ in range(min(num_to_takeoff, len(zggg_airport.departure_queue))):
                    flight_id = zggg_airport.departure_queue.pop(0)
                    f = flight_objects[flight_id]
                    f.status = "In-Flight"
                    f.predicted_departure_time = current_time
                zggg_airport.departure_slot_accumulator -= num_to_takeoff
        
        # 2.4 更新延误时间 (只关心ZGGG出港队列)
        if zggg_airport:
            for flight_id in zggg_airport.departure_queue:
                flight_objects[flight_id].delay_minutes += 1
                flight = flight_objects[flight_id]
                if flight.delay_minutes > 15 and flight.status != "Delayed":
                    flight.status = "Delayed"
                    print(flight)
                    delayed_flights_log.append({
                        "timestamp": current_time,
                        "flight_id": flight.flight_id,
                        "scheduled_departure": flight.scheduled_departure_time,
                        "reason": "Exceeded 15-min wait threshold",
                        "queue_position": zggg_airport.departure_queue.index(flight_id) + 1,
                        "current_queue_size": len(zggg_airport.departure_queue),
                        "airport_capacity" : zggg_airport.current_departure_capacity
                    })
                
        # 2.5 (新增) 每小时结束时记录积压情况
                # 2.5 (升级版) 每小时结束时记录 "延误航班" 的积压情况
        if current_time.minute == 59:
            # 初始化本小时的延误航班计数器
            num_delayed_flights_in_queue = 0
            
            # 只有在机场对象和队列都存在时才进行计算
            if zggg_airport and zggg_airport.departure_queue:
                # 遍历当前队列中的每一个航班ID
                for flight_id in zggg_airport.departure_queue:
                    # 获取航班对象
                    flight = flight_objects[flight_id]
                    # 检查其延误时间是否已经超过15分钟
                    if flight.delay_minutes > 15:
                        num_delayed_flights_in_queue += 1
            
            # 创建日志条目
            log_entry = {
                "timestamp": current_time.replace(minute=0, second=0, microsecond=0),
                # 将列名修改为更能反映其真实含义的名称
                "departure_queue_size": num_delayed_flights_in_queue
            }
            hourly_congestion_log.append(log_entry)
            
        # 2.6 时间前进
        current_time += time_step
        
    print("\n--- 仿真结束 ---")
    
    # --- 步骤三：结果分析 (聚焦于指标2) ---
    results_df = pd.DataFrame([f.__dict__ for f in flight_objects.values()])
    print(results_df)
    log_df = pd.DataFrame(hourly_congestion_log)
    print(log_df)
    # 分析积压时段
    congestion_results = analyze_congestion_periods(log_df, threshold=CONGESTION_THRESHOLD)

    # 找到最晚出港的航班
    departed_flights = results_df[results_df['predicted_departure_time'].notna()]
    latest_departure = departed_flights['predicted_departure_time'].max() if not departed_flights.empty else "无"

    # --- 保存结果到文件 ---
    output_filepath = f"congestion_analysis_report_ZGGG.txt"
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(f"ZGGG机场出港积压分析报告 (仿真结果)\n")
        f.write("="*45 + "\n\n")
        
        # congestion_results 现在总是一个字典，我们检查其内容
        if congestion_results and congestion_results["congestion_periods"]:
            f.write(f"摘要: {congestion_results['summary']}\n")
            for i, period in enumerate(congestion_results["congestion_periods"]):
                f.write(f"\n[积压时段 #{i+1}]\n")
                f.write(f"  - 时段: {period['start_hour'].strftime('%H:%M')} - {period['end_hour'].strftime('%H:%M')}\n")
                f.write(f"  - 持续: {period['duration_hours']} 小时\n")
                f.write(f"  - 峰值: {period['peak_value']} 架 (在 {period['peak_hour'].strftime('%H:%M')} 时)\n")
        else:
            # 如果没有积压时段，也从结果字典中获取摘要信息
            f.write(f"摘要: {congestion_results.get('summary', '当日未监测到积压时段')}\n")
        
        f.write("\n\n")
        f.write("补充项:\n")
        f.write("----------------------------\n")
        latest_dep_str = latest_departure.strftime('%Y-%m-%d %H:%M:%S') if isinstance(latest_departure, pd.Timestamp) else '无'
        f.write(f"  - 当天最晚离港: {latest_dep_str}\n")

    print(f"\n[+] 积压分析报告已保存到: {output_filepath}")
    # 同时保存一份详细的航班日志，以备核查
    results_df.to_csv(f"full_flight_log_ZGGG.csv", index=False)
    print(f"[+] 完整的航班运行日志已保存到: full_flight_log_ZGGG.csv")

    if delayed_flights_log:
        delayed_df = pd.DataFrame(delayed_flights_log)
        delayed_log_filepath = f"delayed_flights_list.csv"
        delayed_df.to_csv(delayed_log_filepath, index=False)
        print(f"[+] 延误航班名单已保存到: {delayed_log_filepath}")
    else:
        print("[+] 本次仿真中没有航班被判定为延误 (延误均未超过15分钟)。")
    
