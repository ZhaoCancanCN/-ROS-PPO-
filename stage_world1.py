import time
import rospy
import copy
import tf
import numpy as np


from geometry_msgs.msg import Twist, Pose  #包含这样的数据类型的数据
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8

#env = StageWorld(512, index=rank, num_env=NUM_ENV)
class StageWorld():
    def __init__(self, beam_num, index, num_env):
        self.index = index  #代表的是当前的进程号
        self.num_env = num_env      #环境的数目
        node_name = 'StageEnv_' + str(index)   #代表的是当前环境的节点名
        rospy.init_node(node_name, anonymous=None)#初始化ros节点  名称都是唯一的    

        self.beam_mum = beam_num  #512线程的光束数目
        self.laser_cb_num = 0   #这是激光的返回数目？？？反正就算设置的一个变量
        self.scan = None  #这个是布尔类型的   代表扫描？还是不扫描？？？？？

        # used in reset_world    #机器人的速度  机器人的目标  
        self.self_speed = [0.0, 0.0]   
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.   #这个代表的是旋转的角度？？？？？

        # used in generate goal point  generate_goal_point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m  创建了一个数组8*8的打印的输出就为【8，8】
        self.goal_size = 0.50  #目标的大小？？

        self.robot_value = 10.   #这个代表的是啥东西嘛self  
        self.goal_value = 0.  
        # self.reset_pose = None

        self.init_pose = None  #初始的为空



        # for get reward and terminate
        self.stop_counter = 0

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)    #实例化 发布者 对象      参数（发布者的名称  、Twist类型的数据、数据可缓存的长度）

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=2)   #实例化 发布者 对象      参数（发布者的名称  、Pose类型的数据、数据可缓存的长度）

        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)#订阅的是基坐标的

        laser_topic = 'robot_' + str(index) + '/base_scan'

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback) #订阅雷达话题

        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)  #订阅里程计信息？？

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)#实例化订阅者对象  判断是否发生了碰撞


        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)  #订阅时间的信息

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)  #服务端的设置


        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None  #这个是当前机器人的速度
        self.state_GT = None
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None:
            pass

        rospy.sleep(1.)#为了休眠1秒钟？？？？
        # # What function to call when you ctrl + c
        # rospy.on_shutdown(self.shutdown)  #现在这里为啥不用了


    def ground_truth_callback(self, GT_odometry):  #这个函数得到了  机器人此刻的位姿和速度  速度是用【线速度,角速度】其中线速度是x 和Y 方向上的矢量和
        Quaternious = GT_odometry.pose.pose.orientation  #四元数的位姿
        Euler = tf.transformations.euler_from_quaternion([Quaternious.x, Quaternious.y, Quaternious.z, Quaternious.w]) #转换为欧拉角
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]  #因为是平面的  所以只需要知道  x   y   和  相应的欧拉角的yaml 偏航角
        v_x = GT_odometry.twist.twist.linear.x    #x方向上的速度
        v_y = GT_odometry.twist.twist.linear.y#y方向上的速度
        v = np.sqrt(v_x**2 + v_y**2)#总的速度方向
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]#速度的偏航角  

    def laser_scan_callback(self, scan):  #激光雷达的数据返回    
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                           scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1  #获取的激光雷达的数目加1 


    def odometry_callback(self, odometry):       #odem 返还的数据   和ground_truth_callback 的区别是啥阿
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]#这个的速度只是含有 x 方向上的速度阿阿   为啥阿

    def sim_clock_callback(self, clock):#返回时间的
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):  #是否发生碰撞   那么这个flag中的data数据就决定了是否发生了碰撞
        self.is_crashed = flag.data

    def get_self_stateGT(self): #获取此刻的状态 
        return self.state_GT

    def get_self_speedGT(self):#获取此刻的速度
        return self.speed_GT

    def get_laser_observation(self): #获得的是激光雷达的数据？？？
        scan = copy.deepcopy(self.scan)   #这里用的的是深拷贝  深拷贝  类似元又创建了一个和其一样的对象  二者的指针完全不同  同时列表里面嵌套的小的列表也是不同的
       #   self.scan = None  #这个是布尔类型的   代表扫描？还是不扫描？？？？？
# 直接赋值：其实就是对象的引用（别名）。
# 浅拷贝(copy)：拷贝父对象，不会拷贝对象的内部的子对象。
# 深拷贝(deepcopy)： copy 模块的 deepcopy 方法，完全拷贝了父对象及其子对象。
        scan[np.isnan(scan)] = 6.0 #np.isnan()是判断是否是空值
        scan[np.isinf(scan)] = 6.0#True if value is inf
# nan：not a number
# inf：infinity;正无穷
# numpy中的nan和inf都是float类型
# 1.np.nan == np.nan  返回 false，同理np.nan != np.nan 返回 true；
# 2.想要判断某元素是否为np.nan可以使用np.isnan(a), a=np.nan, np.isnan(a)返回true；
# 3.np.nan参与任何计算都为np.nan
# np.inf == np.inf  返回 true
# np.inf - np.inf = nan   np.inf + np.inf = np.inf
# 1/np.inf = 0.0
        raw_beam_num = len(scan)  #光束的数量
        sparse_beam_num = self.beam_mum#这个传入的数值为512
        step = float(raw_beam_num) / sparse_beam_num  #这个step代表的是啥子阿    为啥获取的激光数目除以512啊
        sparse_scan_left = []  #左边的稀疏的激光  ？？空的列表 阿
        index = 0.
        for x in xrange(int(sparse_beam_num / 2)):    #这块为啥是Xrange阿   这样怎么循环   
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in xrange(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])   #右边的产生的激光数据 阿    阿阿阿阿   
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)  #列表[::-1]代表的含义是反转列表 从倒数打印    此函数的含义是拼接  竖着拼接
        return scan_sparse / 6.0 - 0.5   #为啥要除以6阿   还减去了0.5
 # 在python的numpy库数据处理中合并的方法有很多，但是想要合并的效率高，且适合大规模数据拼接，能够一次完成多个数组的拼接的只有numpy.concatenate()函数，在合并数组上会比append()更方便，且更有效率
# 1、numpy.concatenate函数
# 主要作用：沿现有的某个轴对一系列数组进行拼接。
# 2、使用语法numpy.concatenate((a1, a2, ...), axis=0, out=None)
# 3、使用参数
# 将具有相同结构的array序列结合成一个array
# a1,a2,...是数组类型的参数，传入的数组必须具有相同的形状。
# axis 指定拼接的方向，默认axis = 0(逐行拼接)(纵向的拼接沿着axis= 1方向)。
# axis=0，拼接方向为横轴，需要纵轴结构相同，拼接方向可以理解为拼接完成后数量发生变化的方向。
# 注：一般axis = 0，就是对该轴向的数组进行操作，操作方向是另外一个轴，即axis=1。

    def get_self_speed(self):    
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):  #产生局部坐标的目标   也就算相对位置吧        把产生的随机目标点在全局的坐标  转换到 以机器人为中心的局部坐标系
        [x, y, theta] = self.get_self_stateGT()  #描述位姿的用的是x ,y  和偏航角
        [goal_x, goal_y] = self.goal_point  #这个是我们的目标点
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]   #返回小车局部坐标系的目标

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.  #这个应该就算初始的角速度？？？？
        self.start_time = time.time() #这个返回的是开始的时间
        rospy.sleep(0.5)


    def generate_goal_point(self):#产生局部坐标点
        [x_g, y_g] = self.generate_random_goal()
        self.goal_point = [x_g, y_g]  #这个是把类的全局的变量 进行了赋值
        [x, y] = self.get_local_goal()#这样 我们就可以直接调用函数得到我们的局部坐标点

        self.pre_distance = np.sqrt(x ** 2 + y ** 2) #代表的相对距离阿  啊啊啊啊    局部坐标的相对位置
        self.distance = copy.deepcopy(self.pre_distance)  #深复制


    def get_reward_and_terminate(self, t):#获得奖励和我们的终止的时间
        terminate = False
        laser_scan = self.get_laser_observation()#获取激光的观测数据
        [x, y, theta] = self.get_self_stateGT()#获取位姿
        [v, w] = self.get_self_speedGT()#获取的是角速度和线速度
        self.pre_distance = copy.deepcopy(self.distance)#获取目标点的相对位置  #啥意思阿    为啥 又他马的复制了一下的 而且二者是相反复制的
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5#这个代表 距离目标的距离   上一时刻的相对距离减去这1时刻的相对距离
        reward_c = 0
        reward_w = 0
        result = 0
        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:  #二者的相对距离 如果二者的相对距离小于0.5证明我们机器人已经到达了目标点阿
            terminate = True
            reward_g = 15  #和论文是一样的都是15
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -15.  #如果发生了膨胀
            result = 'Crashed'

        if np.abs(w) >  1.05:#如果角速度大于1.05  就惩罚
            reward_w = -0.1 * np.abs(w)

        if t > 150:
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w

        return reward, terminate, result  #这三个result的顺序是有着严格的顺序的

    def reset_pose(self):  #新的阶段开始   产生机器人的新的位置
        random_pose = self.generate_random_pose()
        rospy.sleep(0.01)
        self.control_pose(random_pose)  #执行下面的函数
        [x_robot, y_robot, theta] = self.get_self_stateGT()

        # start_time = time.time()
        while np.abs(random_pose[0] - x_robot) > 0.2 or np.abs(random_pose[1] - y_robot) > 0.2:  #目的是为了让新产生的位置和上一时刻的位置距离大  就需要产生新的位姿
            [x_robot, y_robot, theta] = self.get_self_stateGT()
            self.control_pose(random_pose)
        rospy.sleep(0.01)


    def control_vel(self, action):  #发布机器人的运动指令   对于非合作的机器人  这里不就是控制的关键马？？？
        move_cmd = Twist()#这是控制机器人的消息的类型
        move_cmd.linear.x = action[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[1]
        self.cmd_vel.publish(move_cmd)#发布消息cmd_vel代表的是发布者对象 


    def control_pose(self, pose):
        pose_cmd = Pose()
        assert len(pose)==3#assert是断言的意思就算判断   如果pose的传入的数据的长度为3  那么程序就继续执行下去   但是如果不是3  则就会报出assert的错误
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')  #从欧拉角获取得到四元数
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)#cmd_psode  发布位姿的消息

    def generate_random_pose(self):      #这个是随机产生  机器人的位置的函数    返回的是机器人的位置和偏航角
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)
        dis = np.sqrt(x ** 2 + y ** 2)
        while (dis > 9) and not rospy.is_shutdown():
            x = np.random.uniform(-9, 9)
            y = np.random.uniform(-9, 9)
            dis = np.sqrt(x ** 2 + y ** 2)
        theta = np.random.uniform(0, 2 * np.pi)
        return [x, y, theta]       

    def generate_random_goal(self):#产生一个随机的目标点
        self.init_pose = self.get_self_stateGT()#获取机器人的初始的位置和姿态
        x = np.random.uniform(-9, 9)
        y = np.random.uniform(-9, 9)#从这里面产生随机的目标点
        dis_origin = np.sqrt(x ** 2 + y ** 2)  # 判断一下这个坐标是否超过了半径为9 圆外
        dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)  #得到的目标的位置与初始的位置是否大于
        while (dis_origin > 9 or dis_goal > 10 or dis_goal < 8) and not rospy.is_shutdown():   #判断是否目标点在圆外   目标的距离是否大于8并且小于10  如果不满足就继续产生新的目标点
            x = np.random.uniform(-9, 9)
            y = np.random.uniform(-9, 9)
            dis_origin = np.sqrt(x ** 2 + y ** 2)
            dis_goal = np.sqrt((x - self.init_pose[0]) ** 2 + (y - self.init_pose[1]) ** 2)

        return [x, y]#返回目标点的坐标





