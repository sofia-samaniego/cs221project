import subprocess
games = ['SpaceInvaders-v0','MsPacman-v0','Montezuma-v0']
betas = [0.01,0.03,0.1,0.3,1,3,10]

for game in games:
    for beta in betas:
        subprocess.call(['python','im_async_dql.py', game, str(beta)])

