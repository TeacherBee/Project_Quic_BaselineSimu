### 自适应冗余机制策略相关代码 FindRedThreshold
FindRedThreshold_xx.py : 自动测试设定路径参数，但不同丢包率下三种策略的 FCT，用于确定 τ low,τ high。有点问题
--1.FindRedThreshold_sender.py : 发送方，发送seq代替包，并提供超时重传、ackMap机制
--2.FindRedThreshold_receiver.py : 接收方，接收seq回复ack，提供性能统计机制
--3.FindRedThreshold_script.py : 测试脚本，多次测试取平均值
--4.FindRedThreshold_EventDrivenSimu.py：事件仿真形式，加快仿真速度

使用方法：
python sender.py 0.05 none(已弃用)

python FindRedThreshold_script.py(已弃用)

单次仿真(指定丢包率和策略)：
不冗余：python FindRedThreshold_EventDrivenSimu.py 0.05 none
复制冗余(每x个包复制其中y个)：python FindRedThreshold_EventDrivenSimu.py 0.08 replicate_x_y
复制冗余(每x个包生成1个纠错包)：python FindRedThreshold_EventDrivenSimu.py 0.05 xor_x_1
批量脚本仿真：
python FindRedThreshold__EventDrivenScript.py