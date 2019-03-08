#######################################################################
# Copyright (C)                                                       #
# 2016 - 2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)           #
# 2016 Jan Hakenberg(jan.hakenberg@gmail.com)                         #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import pickle

BOARD_ROWS = 3
BOARD_COLS = 3
BOARD_SIZE = BOARD_ROWS * BOARD_COLS

class State:
    def __init__(self):
        '''状态初始化
        棋盘使用 n * n 的数组进行表示
        棋盘中的数字: 1代表先手, -1代表后手下, 0代表该位置无棋子
        
        '''
        
        # 该状态的数组表示
        self.data = np.zeros((BOARD_ROWS, BOARD_COLS))
        # 该状态下的胜利者
        self.winner = None
        # 该状态的哈希值表示
        self.state_hash = None
        # 该状态是否为终结状态
        self.end = None

    def hash(self):
        '''计算状态的哈希值表示

        Returns
        -------
        int
            状态的哈希值表示
        '''

        if self.state_hash is None:
            self.state_hash = 0
            # 哈希值使用三进制表示
            for i in self.data.reshape(BOARD_ROWS * BOARD_COLS):
                if i == -1:
                    i = 2
                self.state_hash = self.state_hash * 3 + i
        return int(self.state_hash)

    def is_end(self):
        '''判断当前状态是否为终结状态.
        如果为终结状态, 同时判断胜利者是谁
        
        Returns
        -------
        bool
            当前状态是否为终结状态
        '''

        if self.end is not None:
            return self.end
        results = []
        # 检查行
        for i in range(0, BOARD_ROWS):
            results.append(np.sum(self.data[i, :]))
        # 检查列
        for i in range(0, BOARD_COLS):
            results.append(np.sum(self.data[:, i]))
        # 检查主对角线
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, i]
        # 检查反对角线
        results.append(0)
        for i in range(0, BOARD_ROWS):
            results[-1] += self.data[i, BOARD_ROWS - 1 - i]
        # 判断胜者
        for result in results:
            # 先手胜
            if result == 3:
                self.winner = 1
                self.end = True
                return self.end
            # 后手胜
            if result == -3:
                self.winner = -1
                self.end = True
                return self.end
        # 检查是否平局
        sum = np.sum(np.abs(self.data))
        if sum == BOARD_ROWS * BOARD_COLS:
            self.winner = 0
            self.end = True
            return self.end
        # 棋盘还未下完
        self.end = False
        return self.end


    def next_state(self, i, j, symbol):
        '''计算当前状态的后继状态
        
        Parameters
        ----------
        i : int
            下一步动作的行坐标
        j : int
            下一步动作的列坐标
        symbol : int
            动作的执行者(1代表先手, -1代表后手)
        
        Returns
        -------
        State
            下一步棋盘的状态
        '''

        new_state = State()
        new_state.data = np.copy(self.data)
        new_state.data[i, j] = symbol
        return new_state

    def print_state(self):
        '''打印状态信息
        
        '''

        for i in range(0, BOARD_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.data[i, j] == 1:
                    token = '*'
                if self.data[i, j] == 0:
                    token = '0'
                if self.data[i, j] == -1:
                    token = 'x'
                out += token + ' | '
            print(out)
        print('-------------')

def get_all_states_impl(current_state, current_symbol, all_states):
    '''生成当前状态的所有后继状态
    
    Parameters
    ----------
    current_state : State
        当前状态
    current_symbol : int
        执棋者(先手为1, 后手为-1)
    all_states : list
        所有可能的棋盘状态
    
    '''

    for i in range(0, BOARD_ROWS):
        for j in range(0, BOARD_COLS):
            # 遍历所有可以放置棋子的位置
            if current_state.data[i][j] == 0:
                # 在此处落子, 生成后继状态
                newState = current_state.next_state(i, j, current_symbol)
                newHash = newState.hash()
                # 根据哈希值判断该状态是否出现过
                if newHash not in all_states.keys():
                    isEnd = newState.is_end()
                    # 将后继状态加入列表
                    all_states[newHash] = (newState, isEnd)
                    # 如果后继状态不是终结状态, 则继续递归生成
                    if not isEnd:
                        get_all_states_impl(newState, -current_symbol, all_states)

def get_all_states():
    '''生成所有可能的棋盘状态
    
    Returns
    -------
    list
        所有可能的棋盘状态
    '''

    # 从先手,空棋盘开始,生成所有可能的棋盘状态
    current_symbol = 1
    current_state = State()
    all_states = dict()
    all_states[current_state.hash()] = (current_state, current_state.is_end())
    get_all_states_impl(current_state, current_symbol, all_states)
    return all_states

all_states = get_all_states()

class Game:
    def __init__(self, player1, player2):
        '''初始化游戏
        
        Parameters
        ----------
        player1 : Player or HumanPlayer
            玩家1
        player2 : Player or HumanPlayer
            玩家2
        
        '''

        self.p1 = player1
        self.p2 = player2
        self.current_player = None
        self.p1_symbol = 1
        self.p2_symbol = -1
        self.p1.set_symbol(self.p1_symbol)
        self.p2.set_symbol(self.p2_symbol)

    def reset(self):
        '''游戏重置
        
        '''

        self.p1.reset()
        self.p2.reset()

    def alternate(self):
        '''两个玩家轮流下棋
        
        '''

        while True:
            yield self.p1
            yield self.p2

    def play(self, print_state=False):
        '''下棋
        
        Parameters
        ----------
        print_state : bool, optional
            是否打印棋盘状态
        
        Returns
        -------
        int
            胜利者
        '''

        alternator = self.alternate()
        # 初始化玩家以及棋盘的状态
        self.reset()
        current_state = State()
        # 将当前状态加入各自的状态列表
        self.p1.set_state(current_state)
        self.p2.set_state(current_state)
        while True:
            # 当前执棋者
            player = next(alternator)
            if print_state:
                current_state.print_state()
            # 落棋位置
            [i, j, symbol] = player.act()
            # 后继状态的哈希值
            next_state_hash = current_state.next_state(i, j, symbol).hash()
            # 根据哈希值查表, 更新状态
            current_state, is_end = all_states[next_state_hash]
            # 各自记录状态
            self.p1.set_state(current_state)
            self.p2.set_state(current_state)
            # 终结状态
            if is_end:
                if print_state:
                    current_state.print_state()
                # 判断胜者
                return current_state.winner

class AgentPlayer:

    def __init__(self, step_size=0.1, epsilon=0.1):
        '''Agent初始化
        
        Parameters
        ----------
        step_size : float, optional
            更新步长
        epsilon : float, optional
            探索概率
        
        '''

        # 值函数
        self.value = dict()
        # 值函数更新步长
        self.step_size = step_size
        # Agent探索概率
        self.epsilon = epsilon
        # Agent在一轮游戏中经历的所有状态
        self.states = []
        # 记录每个状态是否采取贪心策略
        self.greedy = []

    def reset(self):
        '''重置Agent的状态, 开启新一轮游戏
        
        '''

        self.states = []
        self.greedy = []

    def set_state(self, state):
        '''将当前棋盘状态加到Agent的状态列表
        
        Parameters
        ----------
        state : State
            当前棋盘的状态
        
        '''

        self.states.append(state)
        # 默认采用贪心策略
        self.greedy.append(True)

    def set_symbol(self, symbol):
        '''根据先后手, 初始化Agent的值函数
        
        Parameters
        ----------
        symbol : int
            先手还是后手
        
        '''

        self.symbol = symbol
        for state_hash in all_states.keys():
            (state, is_end) = all_states[state_hash]
            if is_end: # 终结状态
                if state.winner == self.symbol: # 获胜
                    self.value[state_hash] = 1.0
                elif state.winner == 0: # 平局
                    self.value[state_hash] = 0.5
                else: # 失败
                    self.value[state_hash] = 0
            else: # 非终结状态
                self.value[state_hash] = 0.5

    def backup(self):
        '''值函数迭代
        
        '''

        # for debug
        # print('player trajectory')
        # for state in self.states:
        #     state.print_state()

        #　获取状态的哈希值表示
        self.states = [state.hash() for state in self.states]
        
        # 逆序遍历所有的状态, 并进行值函数的更新
        for i in reversed(range(len(self.states) - 1)):
            state = self.states[i]  
            # TD误差 = V(s_{t + 1}) - V(s_{t})
            td_error = self.greedy[i] * (self.value[self.states[i + 1]] - self.value[state])
            # TD-Learning(时序差分学习)更新公式
            self.value[state] += self.step_size * td_error

    def act(self):
        '''根据状态采取动作
        
        Returns
        -------
        list
            采取的动作
        '''

        # 获取当前状态
        state = self.states[-1]
        # 下一步所有可能的状态
        next_states = []
        # 下一步所有可能的位置
        next_positions = []
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                # 当前棋盘位置上无棋子
                if state.data[i, j] == 0:
                    # 可行的位置
                    next_positions.append([i, j])
                    # 可行的状态
                    next_states.append(state.next_state(i, j, self.symbol).hash())
        
        # 有epsilon概率采取随机动作
        if np.random.rand() < self.epsilon:
            action = next_positions[np.random.randint(len(next_positions))]
            action.append(self.symbol)
            self.greedy[-1] = False
            return action

        values = []
        # 遍历下一步所有可能的状态和位置
        for state_hash, pos in zip(next_states, next_positions):
            # 获取对应状态的值函数
            values.append((self.value[state_hash], pos))
        # 如果有多个状态的值函数相同,且都是最高的,shuffle则起到在这些状态中随机选择的作用
        np.random.shuffle(values)
        # 按值函数从大到小排序
        values.sort(key=lambda x: x[0], reverse=True)
        # 选取最优动作
        action = values[0][1]
        action.append(self.symbol)
        return action

    def save_policy(self):
        '''保存学习好的值函数(策略)
        
        '''

        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'wb') as f:
            pickle.dump(self.value, f)

    def load_policy(self):
        '''加载保存好的值函数(策略)
        
        '''

        with open('policy_%s.bin' % ('first' if self.symbol == 1 else 'second'), 'rb') as f:
            self.value = pickle.load(f)

# human interface
# input a number to put a chessman
# | q | w | e |
# | a | s | d |
# | z | x | c |
class HumanPlayer:
    def __init__(self, **kwargs):
        self.symbol = None
        self.keys = ['q', 'w', 'e', 'a', 's', 'd', 'z', 'x', 'c']
        self.state = None
        return

    def reset(self):
        return

    def set_state(self, state):
        self.state = state

    def set_symbol(self, symbol):
        self.symbol = symbol
        return

    def backup(self, _):
        return

    def act(self):
        self.state.print_state()
        key = input("Input your position:")
        data = self.keys.index(key)
        i = data // int(BOARD_COLS)
        j = data % BOARD_COLS
        return [i, j, self.symbol]

def train(epochs, print_every_n=500):
    '''对Agent进行训练
    
    Parameters
    ----------
    epochs : int
        训练轮数
    print_every_n : int, optional
        每多少轮输出训练信息
    
    '''

    # 定义两个Agent
    player1 = AgentPlayer(epsilon=0.01)
    player2 = AgentPlayer(epsilon=0.01)
    game = Game(player1, player2)
    # 先手赢的次数
    player1_win = 0.0
    # 后手赢的次数
    player2_win = 0.0
    for i in range(1, epochs + 1):
        # 新的一轮游戏
        game.reset()
        winner = game.play(print_state=False)
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
        # 打印各自的胜率
        if i % print_every_n == 0:
            print('Epoch %d, player 1 winrate: %.02f, player 2 winrate: %.02f' % (i, player1_win / i, player2_win / i))
        # 在每轮游戏结束后,对Agent进行学习
        player1.backup()
        player2.backup()
    # 保存训练好的策略
    player1.save_policy()
    player2.save_policy()

def compete(turns):
    '''将训练好的两个Agent进行对弈
    
    Parameters
    ----------
    turns : int
        对弈轮数
    
    '''

    player1 = AgentPlayer(epsilon=0)
    player2 = AgentPlayer(epsilon=0)
    game = Game(player1, player2)
    player1.load_policy()
    player2.load_policy()
    player1_win = 0.0
    player2_win = 0.0
    for _ in range(0, turns):
        game.reset()
        winner = game.play()
        if winner == 1:
            player1_win += 1
        if winner == -1:
            player2_win += 1
    print('%d turns, player 1 win %.02f, player 2 win %.02f' % (turns, player1_win / turns, player2_win / turns))

def play():
    '''人类玩家和Agent进行对弈
    
    '''

    while True:
        player1 = HumanPlayer()
        player2 = AgentPlayer(epsilon=0)
        game = Game(player1, player2)
        player2.load_policy()
        winner = game.play()
        if winner == player2.symbol:
            print("失败!")
        elif winner == player1.symbol:
            print("胜利!")
        else:
            print("平局!")

if __name__ == '__main__':
    train(int(1e5))
    compete(int(1e3))
    play()


