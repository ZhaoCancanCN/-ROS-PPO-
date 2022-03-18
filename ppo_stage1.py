import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI  #定义之后即可完成初始化

from torch.optim import Adam
from collections import deque   #这个是双向队列

from model.net import MLPPolicy, CNNPolicy
from stage_world1 import StageWorld
from model.ppo import ppo_update_stage1, generate_train_data
from model.ppo import generate_action
from model.ppo import transform_buffer



MAX_EPISODES = 5000  #Tmax这个是训练多久？？？？？
LASER_BEAM = 512#这个代表的是激光雷达获取180度的数据个数
LASER_HIST = 3 #激光雷达的数据选的是三层
HORIZON = 128#眼界   看的最远的距离？？？？
GAMMA = 0.99  #损失伽马r
LAMDA = 0.95#人
BATCH_SIZE = 1024
EPOCH = 2   #这个和深度学习有关
COEFF_ENTROPY = 5e-4  #什么熵  系数？？？
CLIP_VALUE = 0.1#这个和PPO算法有关
NUM_ENV = 24    #环境的个数
OBS_SIZE = 512    #激光雷达观察到的数据
ACT_SIZE = 2
LEARNING_RATE = 5e-5#这是第一阶段学习率



def run(comm, env, policy, policy_path, action_bound, optimizer):#这个comm感觉有点类似self  不对阿   self  不是那个类里面的定义马？？

    # rate = rospy.Rate(5)
    buff = []      #定义的缓存replaybuffer是一个空的列表
    global_update = 0
    global_step = 0


    if env.index == 0:#index代表的是环境的
        env.reset_world()#这里的重置环境代表的沙意思


    for id in range(MAX_EPISODES):
        env.reset_pose()

        env.generate_goal_point()
        terminal = False
        ep_reward = 0
        step = 1

        obs = env.get_laser_observation()#获取激光雷达的数据
        obs_stack = deque([obs, obs, obs])  #这里的obs获取的激光雷达的数据  为啥是三个  deque代表的是一个队列的意思  stack代表的堆栈的意思，因为时间是第一次开始的，所以三第一次获取的前三贞全是这个？
        goal = np.asarray(env.get_local_goal())#目标点 这个是相对位置   这个是将数组或者列表转换为矩阵  如果原来的数据是数组  那么asarray会随着元数据的改变而自动改变
        speed = np.asarray(env.get_self_speed())#获取机器人自身的速度
        state = [obs_stack, goal, speed]#机器人的状态记作激光雷达的数据    目标点   速度

        while not terminal and not rospy.is_shutdown():  #判断是否roscore关闭了
            state_list = comm.gather(state, root=0)#状态列表 状态的聚集


            # generate actions at rank==0   当前的进程号
            v, a, logprob, scaled_action=generate_action(env=env, state_list=state_list,
                                                         policy=policy, action_bound=action_bound)   #这个是来返回速度 a 这个和PPO算法相关了

            # execute actions   这里貌似只有这个 scaled_action起作用了
            real_action = comm.scatter(scaled_action, root=0)#这个scatter又代表啥意思？？？？将得到的动作列表数据都分发出去？？
            env.control_vel(real_action)     #这个速度怎么控制的还不知道，在stageworld里面有介绍的

            # rate.sleep() 频率  1s1000次
            rospy.sleep(0.001)  

            # get informtion   
            r, terminal, result = env.get_reward_and_terminate(step) #这个是获取的执行完的汇报step为1阿
            ep_reward += r   #reward为啥没有进行一个折扣呢？  不对这个是这个时刻的奖励
            global_step += 1#步长加1


            # get next state
            s_next = env.get_laser_observation()
            left = obs_stack.popleft()   #popleft 函数的作用也就是将左边的变量拿出，也就是将最新的激光数据加进去 最晚的数据拿出来
            obs_stack.append(s_next)    #加入最新数据
            goal_next = np.asarray(env.get_local_goal())#这个get_local_goal应该是不变的
            speed_next = np.asarray(env.get_self_speed())
            state_next = [obs_stack, goal_next, speed_next]#下一时刻的状态和速度

            if global_step % HORIZON == 0:   #这不是取余数吗？也就算为了得到终止状态的下一个状态？？？128
                state_next_list = comm.gather(state_next, root=0)#这个comm 的概念不是很懂阿，这个是多个环境下一时刻的所有的状态的列表集合吗？
                last_v, _, _, _ = generate_action(env=env, state_list=state_next_list, policy=policy,
                                                               action_bound=action_bound)#这里我们实际上也只是需要一个最新的V，这个也就是我们最长步128时刻
            # add transitons in buff and update policy 都有一个root=0   不懂啥意思
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)#terminal  代表的是终止？？还是是否为终止 这里面的 都是由并行运行得到的 gather得到的

            if env.index == 0:
                buff.append((state_list, a, r_list, terminal_list, logprob, v))  #状态列表包含3个元素   其他的按照顺序依次类推
                if len(buff) > HORIZON - 1:  #如果超过了buff的长度超过了我的视野
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)   #这样通过这个transform_buffer函数我们就可以   得到一批的训练数据了  这个在PPO中有的定义
                    t_batch, advs_batch = generate_train_data(rewards=r_batch, gamma=GAMMA, values=v_batch,
                                                              last_value=last_v, dones=d_batch, lam=LAMDA)  #d_batch代表的是终止时间的节点
                                                              # t_batch, 代表我们的目标 advs_batch代表目标与实际的差值？？？
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    #进行网络的训练  
                    ppo_update_stage1(policy=policy, optimizer=optimizer, batch_size=BATCH_SIZE, memory=memory,
                                            epoch=EPOCH, coeff_entropy=COEFF_ENTROPY, clip_value=CLIP_VALUE, num_step=HORIZON,
                                            num_env=NUM_ENV, frames=LASER_HIST,
                                            obs_size=OBS_SIZE, act_size=ACT_SIZE)

                    buff = []#重新设置我们buff为空   开始下一轮的数据采集 
                    global_update += 1

            step += 1
            state = state_next


        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:#0时刻模型不保存  以后每20保存一次
                #1) state_dict是在定义了model或optimizer之后pytorch自动生成的,可以直接调用.常用的保存state_dict的格式是".pt"或'.pth'的文件,即下面命令的 PATH="./***.pt"
                #torch.save(model.state_dict(), PATH)
                torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                #/Stage1_{}  中的{},最终可以被global_update的数值代替  policy在函数的使用的时候作为参数调用
                #pytorch中torch.save()和torch.load()分别表示模型的存储和加载，具体相关的用法如下所示：
               # torch.save(obj,f,pickle_module=pickle,pickle_protocol=DEFAULT_PROTOCOL,_use_new_zipfile_serialization=True)
                # obj：需要保存的对象，可以是整个模型或者是模型参数
                # f：保存模型的路径
                # pickle_module：用于清除元数据和对象的模块
                # pickle_protocol：可以指定覆盖默认协议
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))#这个是打印输出的
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)#坐标x,y

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, setp %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)  #这个就把reward数据保存了下来了





if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()#这里的socket的用处也仅仅就算为了获得主机的名字
    if not os.path.exists('./log/' + hostname):#是否存在这个log文件
        os.makedirs('./log/' + hostname)#就创建了这个文件
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'  #这个就是存放的rewad的数值

    # config log
    logger = logging.getLogger('mylogger')#这个logger 
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')#这里告诉了存储的位置  output_file 
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')#这个代表的是logger_cal,输出的cal_file  也告诉了位置
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank()#获取当前进程号
    size = comm.Get_size()#获取通信域的进程数

    env = StageWorld(512, index=rank, num_env=NUM_ENV)#这里是改环境个数的地方
    reward = None
    action_bound = [[0, -1], [1, 1]]

    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)#这里要用到卷集神经网络
        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)#优化器选择的是Adam，优化器的作用和好处还不知道，SGB 随机梯度下降很多方法
        mse = nn.MSELoss()#这里采用的是损失函数

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/stage1_2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)#这一句话是用以前训练过的模型的
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, policy_path=policy_path, action_bound=action_bound, optimizer=opt)#这个是程序的入口
    except KeyboardInterrupt:
        pass