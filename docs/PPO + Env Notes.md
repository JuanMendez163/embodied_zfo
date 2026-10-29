**PPO NOTES**
Every dt timestep:
- agent picks a new action (move_cmd, turn_cmd, eat_cmd)
- agent attempts to eat
- glucose is updated, including decay over dt minus movement cost plus an intake amount
- reward is computed following the glucose change
- new observation formed for that complete state, which agent uses for next action

**Eat bouts**: runs of dt when the agent consecutively eats
- used to compute possible_eats, which provides an upper bound on the max *possible* food pellets the agent could have eaten given the eat commands it did
	- It is not a tight bound because it assumes the cooldown is over when the bout starts. This may not actually be true
	- Note: this metric penalizes eating when the cooldown is not over
- We use this value to then compute eat_efficiency, which compares the total actual eaten vs. the upper bound on eaten
- Includes a "tap" eat, that is just one dt of eat

evaluate() returns a dict with all the episode_metrics metrics as keys, and the values for each evaluation episode as values
- every evaluation episode has a different reset seed, but across policies the seeds are the same (every policy calls the same evaluate() function, so we can compare across policies)

General notes about PPO: during training, the Actor network outputs a Gaussian distribution over actions. Actions are randomly sampled from that distribution, allowing for exploration + some gradient stuff I don't follow understand. During evaluation, we don't take random actions over the distribution: *we just take the mean*, which represents the policy's best guess of what to do. During training, the Gaussian distribution's standard deviation shifts and is learned, and represents the policy's confidence of what to do for that state.

In PPO, every observation gets mapped to a mean for a Gaussian distribution. Observations share the same standard deviation, which represents how much the network is willing to do exploratory actions. Actions are randomly sampled from the Gaussian. Standard deviation updates according to whether better outcomes go with bigger or smaller deviations from the mean.
- if being near mean tends to do better (A > 0), shrink standard deviation.

**Horizon/Gamma:**
- **Horizon:** tells you how far into the future the agent "cares" about (i.e. affects the value of that state). Measured in seconds
- **Gamma:** The same concept, but expressed per step rather than seconds. Each step into the future, a reward is multiplied by gamma once more (with an exponential decay of future step importance). Gamma is derived from the horizon, and is derived in such a way for discounts to be dt-independent (to prevent smaller dt's from having more gamma impact)
	- The exponential decay is scaled over horizon according to dt

make_ppo_env(): RescaleAction maps PPO distribution action outputs (default start is N(0,1)) to the action space of the commands. Relevant for move_cmd for instance, which is in (0,1); without the rescale, all negative PPO outputs would get clipped to 0 and that information would be ignored in actually shaping actions

Note: a number of times the logger stores means of episode metrics across rollouts and such; these should largely be taken with caution because not every episode metric is amenable to means. Every single episode's episode_metrics are also stored, and we will likely use those for later training and evaluation analysis

**PPO Architecture**:
- Actor-Critic networks
- Actor: takes actions given a state
- Critic: estimates value of different actions for each state
- Each are MLPs with two hidden layers, each layer having 64 units and tanh activations

**Training Loop:**
1. Collect a rollout: N_envs x n_steps gives you a certain number of transitions. Policy stays fixed during this collection
2. Per transition, two numbers are computed: Return R (the discounted sum of rewards that followed this step), and Advantage A (R-V(s); how much better this action turned out than the critic expected)
	1. gae_lambda: a parameter that controls how advantage is computed. Sets the tradeoff between trusting the actual rewards and trusting the critic
3. Learn from it: n_epochs passes through all the transitions. Split the transitions into minibatches of size batch_size, and take one gradient step on each minibatch
	1. A gradient step is one update of the Actor and Critic networks
	2. Policy losses:
		1. Actor: "make good actions more likely." Basing off A, change the policy towards actions that led to a higher advantage (better returns than expected). Updates are clipped towards small updates
		2. Value: Update your value estimates based off the difference with R, the actual discounted sum of rewards following that transition
		3. The relationship between Actor and Critic is only through advantage; actions that perform better than expected (expected coming from critic) have higher probability of selection



**ENV NOTES**
- We scale reward to put in order 1 (good for PPO)
- Starting glucose can be a random number from (low, high) range
- **Observation**: **(glucose/glucose_max, can_eat, proximity, cos(theta), sin(theta), in_reach)**
	- proximity: 1 - (d/range), d is Euclidean distance to nearest food, range is sensing range (default arena diagonal, i.e. can sense entire arena)
		- Why we use proximity: proximity=0 represents the weakest signal, while distance=0 represents the strongest signal. Distance's value has an inverse relationship to the importance to the network, which messes up PPO. A proximity of 0 corresponds to "nothing there," which corresponds to no signal in the network.
		- Value rises as food gets closer
	- theta: angle of nearest food relative to current heading, in range (-pi, pi). Egocentric direction
		- We take cos(theta), sin(theta) to have a smooth representation of nearest food direction (i.e. angles near pi and -pi are actually quite similar directions but radically different values; taking cos/sin ensures a consistent representation)
	- You can only sense food within sensing_range: relates to distance, i.e. can only sense food with distance < sensing_range
- **Step:**
	- 1. Store current glucose
	- 2. Try eating (depends on cooldown, mouth in reach, and eat_cmd)
	- 3. Move (depends on move_cmd and turn_cmd)
	- 4. Update glucose, update time_since_last_eat
	- 5. Compute reward
	- 6. Update step count, check for starvation / time limit, return observation
- Glucose half life: the time in seconds it takes for glucose to halve with no food and swimming. This sets the exponential decay constant for the glucose dynamics. A value to research in the literature
- ![[glucose_dynamics.png]]
- 