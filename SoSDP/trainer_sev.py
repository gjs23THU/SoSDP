import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim

from tqdm.auto import tqdm

from SoS_env import SoSEnv
from model_sev import DRL4SoSEnvActor, StateCritic


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _obs_batch_to_tensors(obs_batch):
    static = torch.from_numpy(np.stack([obs['static'] for obs in obs_batch], axis=0)).float().to(device)
    dynamic = torch.from_numpy(np.stack([obs['dynamic'] for obs in obs_batch], axis=0)).float().to(device)
    mask = torch.from_numpy(np.stack([obs['mask'] for obs in obs_batch], axis=0)).float().to(device)
    return static, dynamic, mask


def _make_envs(batch_size, args, seed_offset=0, num_samples=None):
    samples = int(num_samples) if num_samples is not None else int(args.train_size)
    envs = []
    for i in range(batch_size):
        envs.append(
            SoSEnv(
                num_systems=args.num_nodes,
                num_samples=samples,
                num_caps=args.capnum,
                seed=args.seed + seed_offset + i,
            )
        )
    return envs


def _reset_envs(envs, seed_base=None):
    obs_batch = []
    for i, env in enumerate(envs):
        if seed_base is None:
            obs, _ = env.reset()
        else:
            obs, _ = env.reset(seed=seed_base + i)
        obs_batch.append(obs)
    return obs_batch


def rollout_batch(actor, envs, max_steps):
    obs_batch = _reset_envs(envs)

    static0, dynamic0, _ = _obs_batch_to_tensors(obs_batch)
    actor_state = actor.init_episode(static0, dynamic0)

    batch_size = len(envs)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    sum_logp = torch.zeros(batch_size, dtype=torch.float32, device=device)
    returns = torch.zeros(batch_size, dtype=torch.float32, device=device)
    lengths = torch.zeros(batch_size, dtype=torch.float32, device=device)

    for _ in range(max_steps):
        if torch.all(done):
            break

        _, _, mask = _obs_batch_to_tensors(obs_batch)

        # Keep finished episodes numerically stable during batched sampling.
        safe_mask = mask.clone()
        safe_mask[done] = 0.0
        if safe_mask.size(1) > 0:
            safe_mask[done, 0] = 1.0

        action, logp, actor_state, _ = actor.forward_step(actor_state, safe_mask, done_mask=done)
        actions = action.detach().cpu().numpy()

        next_obs_batch = list(obs_batch)
        for i, env in enumerate(envs):
            if done[i].item():
                continue

            obs, reward, terminated, truncated, _info = env.step(int(actions[i]))
            next_obs_batch[i] = obs

            returns[i] += float(reward)
            lengths[i] += 1.0

            if terminated or truncated:
                done[i] = True

        sum_logp += logp
        obs_batch = next_obs_batch

    return {
        'static0': static0,
        'dynamic0': dynamic0,
        'returns': returns,
        'sum_logp': sum_logp,
        'mean_len': lengths.mean().item(),
    }


def validate_env(actor, args, episodes=128):
    actor.eval()

    batch_size = min(args.batch_size, episodes)
    envs = _make_envs(batch_size, args, seed_offset=100000, num_samples=args.valid_size)

    total_returns = []
    with torch.no_grad():
        while len(total_returns) < episodes:
            batch = rollout_batch(actor, envs, max_steps=envs[0].max_steps)
            total_returns.extend(batch['returns'].detach().cpu().tolist())

    for env in envs:
        env.close()

    actor.train()
    return float(np.mean(total_returns[:episodes]))


def train_sos_env(args):
    static_size = 13
    dynamic_size = 1

    actor = DRL4SoSEnvActor(
        static_size=static_size,
        dynamic_size=dynamic_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    critic = StateCritic(static_size, dynamic_size, args.hidden_size).to(device)

    if args.checkpoint:
        actor_path = os.path.join(args.checkpoint, 'actor.pt')
        critic_path = os.path.join(args.checkpoint, 'critic.pt')
        actor.load_state_dict(torch.load(actor_path, map_location=device))
        critic.load_state_dict(torch.load(critic_path, map_location=device))

    actor_optim = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optim = optim.Adam(critic.parameters(), lr=args.critic_lr)

    now = datetime.datetime.now().strftime('%H_%M_%S')
    save_dir = os.path.join('SoS_env', str(args.num_nodes), now)
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    envs = _make_envs(args.batch_size, args, seed_offset=0, num_samples=args.train_size)

    best_valid = -np.inf
    updates_per_epoch = max(1, args.train_size // max(1, args.batch_size))

    for epoch in range(args.epochs):
        actor.train()
        critic.train()

        losses = []
        rewards = []
        lengths = []

        epoch_dir = os.path.join(checkpoint_dir, str(epoch))
        os.makedirs(epoch_dir, exist_ok=True)

        start_time = time.time()
        train_iter = tqdm(range(updates_per_epoch), desc=f'Epoch {epoch + 1}/{args.epochs}', leave=False)
        for update in train_iter:
            batch = rollout_batch(actor, envs, max_steps=envs[0].max_steps)

            returns = batch['returns']
            critic_est = critic(batch['static0'], batch['dynamic0']).view(-1)
            advantage = returns - critic_est

            actor_loss = torch.mean(advantage.detach() * batch['sum_logp'])
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            critic_optim.step()

            losses.append(float(actor_loss.detach().cpu().item()))
            rewards.append(float(returns.mean().detach().cpu().item()))
            lengths.append(float(batch['mean_len']))

            train_iter.set_postfix(
                reward=f'{rewards[-1]:.3f}',
                loss=f'{losses[-1]:.4f}',
                len=f'{lengths[-1]:.1f}',
            )

            if (update + 1) % args.log_interval == 0:
                step_actor_path = os.path.join(epoch_dir, f'actor_{update + 1}.pt')
                step_critic_path = os.path.join(epoch_dir, f'critic_{update + 1}.pt')
                torch.save(actor.state_dict(), step_actor_path)
                torch.save(critic.state_dict(), step_critic_path)

        epoch_actor = os.path.join(epoch_dir, 'actor.pt')
        epoch_critic = os.path.join(epoch_dir, 'critic.pt')
        torch.save(actor.state_dict(), epoch_actor)
        torch.save(critic.state_dict(), epoch_critic)

        mean_loss = float(np.mean(losses)) if losses else 0.0
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        mean_len = float(np.mean(lengths)) if lengths else 0.0
        mean_valid = validate_env(actor, args, episodes=args.valid_episodes)

        if mean_valid > best_valid:
            best_valid = mean_valid
            torch.save(actor.state_dict(), os.path.join(save_dir, 'actor.pt'))
            torch.save(critic.state_dict(), os.path.join(save_dir, 'critic.pt'))

        print(
            f'Epoch {epoch + 1}: loss={mean_loss:.4f}, reward={mean_reward:.4f}, '
            f'len={mean_len:.2f}, valid={mean_valid:.4f}, took={time.time() - start_time:.1f}s'
        )

    for env in envs:
        env.close()


def build_parser():
    parser = argparse.ArgumentParser(description='SoS Gym trainer (original RL objective)')
    parser.add_argument('--seed', default=1234567, type=int)
    parser.add_argument('--capnum', default=5, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--nodes', dest='num_nodes', default=30, type=int)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2.0, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=10000, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)

    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--valid-episodes', default=128, type=int)
    parser.add_argument('--log-interval', default=100, type=int)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    train_sos_env(args)
