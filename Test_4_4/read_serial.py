import serial
import struct
import time

def main():
    # 串口配置，请根据实际情况修改端口号（如 COM10）和波特率
    PORT = 'COM10'
    BAUDRATE = 460800
    
    print(f"正在打开串口 {PORT}，波特率 {BAUDRATE}...")
    try:
        # 打开串口，设置超时时间为 0.5 秒，方便快速检测和重试
        ser = serial.Serial(PORT, BAUDRATE, timeout=0.5)
        # 【关键点】打开串口后必须延时 1.5 秒！
        # 很多 USB 转串口芯片在刚打开时会引起 MCU 复位或处于短暂不稳定状态，此时发送数据会丢失。
        print("等待串口线电平稳定（1.5秒）...")
        time.sleep(1.5)
    except Exception as e:
        print(f"打不开串口 {PORT}: {e}")
        print("提示：请确认串口号是否正确，且未被其他串口助手占用。")
        return

    print("串口已成功打开！正在等待接收 4x4 压力矩阵数据...")
    print("提示：按 Ctrl+C 可退出程序。")
    
    # 首次尝试发送启动指令
    try:
        print("向坐垫发送启动指令 '1'...")
        ser.write(b'1')
    except Exception as e:
        print(f"发送启动指令失败: {e}")

    last_send_time = time.time()
    no_data_count = 0

    try:
        while True:
            # 1. 寻找帧头 0xAA 0x55 (对应 STM32 发送的 0x55AA)
            b = ser.read(1)
            if not b:
                # 读超时，说明当前没有收到数据
                no_data_count += 1
                # 如果连续 4 次超时（约 2 秒无数据输入），尝试自动重发启动指令 '1'
                if no_data_count >= 4:
                    current_time = time.time()
                    if current_time - last_send_time > 2.0:
                        print("没有检测到数据流，正在尝试重新发送启动指令 '1'...")
                        try:
                            ser.write(b'1')
                        except:
                            pass
                        last_send_time = current_time
                        no_data_count = 0
                continue
            
            # 只要收到任意数据，就重置无数据计数器
            no_data_count = 0

            if b[0] == 0xAA:
                b2 = ser.read(1)
                if b2 and b2[0] == 0x55:
                    # 2. 找到帧头后，读取接下来的 32 字节数据负载（16 个 uint16_t 点）
                    payload = ser.read(32)
                    if len(payload) == 32:
                        # 3. 将 32 字节解析为 16 个无符号短整型（小端格式）
                        data = struct.unpack('<16H', payload)
                        
                        # 4. 格式化打印数据矩阵
                        current_time = time.strftime('%H:%M:%S')
                        print(f"\n[{current_time}] 收到 4x4 数据帧:")
                        print("----------------------------")
                        for r in range(4):
                            row_vals = data[r*4 : (r+1)*4]
                            # 格式化对齐打印，每个数据点占 6 位字符宽度
                            print("  ".join(f"{val:6d}" for val in row_vals))
                        print("----------------------------")
                        
    except KeyboardInterrupt:
        print("\n正在退出程序...")
        # 尝试发送 '2' 命令停止坐垫传输
        try:
            print("发送停止指令 '2'...")
            ser.write(b'2')
            time.sleep(0.1)
        except:
            pass
    finally:
        ser.close()
        print("串口已关闭。")

if __name__ == '__main__':
    main()
