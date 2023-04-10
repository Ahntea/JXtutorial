import gymnasium as gym
import time

env = gym.make('MountainCarContinuous-v0',render_mode='human') #강화학습 환경 불러오기 MountainCarContinuous-v0 CartPole-v1

for i_episode in range(20):
    # 새로운 에피소드(initial environment)를 불러온다(reset)
    observation = env.reset()
    for t in range(10000):
        env.render() #화면에 출력 / 행동 취하기 이전 환경에서 얻은 관찰값(obsevation)적용해서 그림
        # time.sleep(0.01)

        # 행동(action)을 취하기 이전에 환경에 대해 얻은 관찰값(observation)
        # print('observation before action:')
        # print(observation)
        action = env.action_space.sample()#임의의 action 선택
        obs, reward, terminated, truncated, info = env.step(action)#선택한 action을 환경으로 보냄
        # time.sleep(0.05)

        # 행동(action)을 취한 이후에 환경에 대해 얻은 관찰값(observation)
        # print('observation after action:')
        # print(observation)

        if terminated:
            print("Episode-{} finished after {} timesteps".format(i_episode,t+1))
            
            break
env.close()