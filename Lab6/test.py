'''DLP DQN Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from atari_wrappers import wrap_deepmind, make_atari, LazyFrames
from matplotlib import pyplot as plt
from PIL import Image


class ReplayMemory(object):
    ## TODO ##
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        """Saves a transition"""
        # print(type(transition))
        # self.buffer.append(transition)
        # self.buffer.append(tuple(map(tuple, transition)))
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        """Sample a batch of transitions"""
        transitions = random.sample(self.buffer, batch_size)
        # print(type(transitions[0][3]))
        # print(len(transitions[0]))
        # print(transitions[0][0].__array__().shape)
        # for x in zip(transitions):
        #     print(x)
        #     break
        # # print(zip(transitions))
        # return (x for x in zip(transitions))
        # map(__array__, x)
        # return (x if i == 0 or i == 3 else x for i, x in enumerate(zip(*transitions)))
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(np.asarray(x), dtype=torch.float, device=device) for x in zip(*transitions))



class Net(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super(Net, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=(5,4), stride=2, padding=(2,0)),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # print('before conv', x.shape)
        x = x.float() / 255.
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        # print('after conv', x.shape)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


class DQN:
    def __init__(self, args):
        self._behavior_net = Net().to(args.device)
        self._target_net = Net().to(args.device)
        # initialize target network
        self._target_net.load_state_dict(self._behavior_net.state_dict())
        self._target_net.eval()
        self._optimizer = torch.optim.Adam(self._behavior_net.parameters(), lr=args.lr, eps=1.5e-4)

        ## TODO ##
        """Initialize replay buffer"""
        #self._memory = ReplayMemory(...)
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.freq = args.freq
        self.target_freq = args.target_freq

    def select_action(self, state, epsilon, action_space):
        '''epsilon-greedy based on behavior network'''
        ## TODO ##
        if random.random() < epsilon:
            return action_space.sample()
        else:
            with torch.no_grad():
                # print('state', state.shape)
                
                state_tensor = torch.tensor(state, device=self.device).reshape(1, 72, 84, 4)
                # for i in range(4):
                #     plt.imshow(state_tensor[0,:,:,i].cpu().numpy().astype(np.uint8), cmap='gray')
                #     plt.show()
                # print(state.shape)
                state_tensor = state_tensor.permute((0, 3, 1, 2))

                actions = self._behavior_net(state_tensor)
                return torch.max(actions, dim=1)[1].item()

    def append(self, state, action, reward, next_state, done):
        ## TODO ##
        """Push a transition into replay buffer"""
        #self._memory.push(...)
        # print('state', type(state))
        # print('action', type(action))
        # print('reward', type(reward))
        # print('next_state', type(next_state))
        # print('done', type(done))
        self._memory.append(state, [action], [reward / 10], next_state, [int(done)])

    def update(self, total_steps):
        if total_steps % self.freq == 0:
            self._update_behavior_network(self.gamma)
        if total_steps % self.target_freq == 0:
            self._update_target_network()

    def _update_behavior_network(self, gamma):
        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(self.batch_size, self.device)

        # for k in range(4):
        #     for i in range(72):
        #         for j in range(84):
        #             print(int(state[0, i, j, k].cpu().numpy()), end = ' ')
        #         print('next')


        state = state.permute((0, 3, 1, 2))
        next_state = next_state.permute((0, 3, 1, 2))
        # for i in range(4):
        #     plt.imshow(state[0,i].cpu().numpy().astype(np.uint8), cmap='gray')
        #     plt.show()
        # print(state.shape)


        # print('state', state.shape)
        # print(len(state))
        # print('action', type(action))
        # print('reward', type(reward))
        # print('next_state', type(next_state))
        # print('done', type(done))
        ## TODO ##
        q_value = self._behavior_net(state).gather(dim=1, index=action.long())
        with torch.no_grad():
            q_next = self._target_net(next_state)
            max_q_next = torch.max(q_next, dim=1)[0].reshape(-1, 1)
            q_target = reward + gamma*max_q_next*(1-done)
        criterion = nn.MSELoss()
        loss = criterion(q_value, q_target)
        # optimize
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._behavior_net.parameters(), 5)
        self._optimizer.step()

    def _update_target_network(self):
        '''update target network by copying from behavior network'''
        ## TODO ##
        self._target_net.load_state_dict(self._behavior_net.state_dict())

    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
                'target_net': self._target_net.state_dict(),
                'optimizer': self._optimizer.state_dict(),
            }, model_path)
        else:
            torch.save({
                'behavior_net': self._behavior_net.state_dict(),
            }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._behavior_net.load_state_dict(model['behavior_net'])
        if checkpoint:
            self._target_net.load_state_dict(model['target_net'])
            self._optimizer.load_state_dict(model['optimizer'])


def train(args, agent, writer):
    print('Start Training')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, frame_stack=True, clip_rewards=True, episode_life=True)
    action_space = env.action_space
    total_steps, epsilon = 0, 1.
    ewma_reward = 0

    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        # print(state.__array__().shape)
        state, reward, done, _ = env.step(1) # fire first !!!
        state = state.__array__()[12:, :, :]
        # print(state.shape)
        # for k in range(4):
        #     for i in range(84):
        #         for j in range(84):
        #             print(state[i, j, k], end =" ")
        #         print('next')
        # for i in range(4):
        #     image = Image.fromarray(state[:,:,i])
        #     image.show()
        # print(state.shape)
        for t in itertools.count(start=1):
            if total_steps < args.warmup:
                action = action_space.sample()
            else:
                # select action
                action = agent.select_action(state, epsilon, action_space)
                # decay epsilon
                epsilon -= (1 - args.eps_min) / args.eps_decay
                epsilon = max(epsilon, args.eps_min)

            # execute action
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.__array__()[12:, :, :]
            ## TODO ##
            # store transition
            #agent.append(...)
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                agent.update(total_steps)

            state = next_state
            total_reward += reward
            total_steps += 1

            if total_steps % args.eval_freq == 0:
                """You can write another evaluate function, or just call the test function."""
                test(args, agent, writer, episode)
                agent.save(args.model + "dqn_" + str(total_steps) + ".pt")

            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward, episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, episode)
                print('Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}\tEpsilon: {:.3f}'
                        .format(total_steps, episode, t, total_reward, ewma_reward, epsilon))
                break
    env.close()


def test(args, agent, writer, episode):
    print('Start Testing')
    env_raw = make_atari('BreakoutNoFrameskip-v4')
    env = wrap_deepmind(env_raw, frame_stack=True, clip_rewards=False, episode_life=False)
    action_space = env.action_space
    e_rewards = []
    
    for i in range(args.test_episode):
        state = env.reset()
        e_reward = 0
        done = False
        counter = 0
        while not done:
            time.sleep(0.01)
            env.render()

            if counter > 200:
                epsilon = (counter-200)*0.001
            else:
                epsilon = 0
            action = agent.select_action(state.__array__()[12:, :, :], epsilon, action_space)
            state, reward, done, _ = env.step(action)
            e_reward += reward
            if reward == 0:
                counter = counter + 1
            else:
                counter = 0
            

        print('episode {}: {:.2f}'.format(i+1, e_reward))
        e_rewards.append(e_reward)

    env.close()
    if episode != 0:
        writer.add_scalar('Test/Average Reward', float(sum(e_rewards)) / float(args.test_episode), episode)
    print('Average Reward: {:.2f}'.format(float(sum(e_rewards)) / float(args.test_episode)))


def main():
    ## arguments ##
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--device', default='cuda')
    parser.add_argument('-m', '--model', default='ckpt/crop_little_long/')
    parser.add_argument('--logdir', default='log/dqn_breakout_life_crop_little_long')
    # train
    parser.add_argument('--warmup', default=20000, type=int)
    parser.add_argument('--episode', default=150000, type=int)
    parser.add_argument('--capacity', default=100000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.0000625, type=float)
    parser.add_argument('--eps_decay', default=1000000, type=float)
    parser.add_argument('--eps_min', default=0.1, type=float)
    parser.add_argument('--gamma', default=.99, type=float)
    parser.add_argument('--freq', default=4, type=int)
    parser.add_argument('--target_freq', default=10000, type=int)
    parser.add_argument('--eval_freq', default=200000, type=int)
    # test
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('-tmp', '--test_model_path', default='ckpt/crop_little_long/dqn_6000000.pt')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--test_episode', default=10, type=int)
    parser.add_argument('--seed', default=20230422, type=int)
    parser.add_argument('--test_epsilon', default=0.01, type=float)
    args = parser.parse_args()

    ## main ##
    
    agent = DQN(args)
    # agent.load(args.test_model_path)
    writer = SummaryWriter(args.logdir)
    if args.test_only:
        agent.load(args.test_model_path)
        test(args, agent, writer, episode=0)
    else:
        train(args, agent, writer)
        


if __name__ == '__main__':
    main()
