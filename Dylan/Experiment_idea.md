# Goals:

Predator Focused (experimental block 1)

- Compare predator performance between PPO and MAPPO algorithms (with parameter sharing in both). Comparison with baseline random algorithm as well. Trained on random prey
- Observe (visual) emergent behaviour by running the simulation
- Investigate real-life hunting scenarios by introducing reward shaping (ie: movement penalties, proximity reward, teamwork reward )

Prey Focused (experimental block 2)

- Compare prey performance between PPO and MAPPO algorithms (with parameter sharing in both). Comparison with baseline random algorithm as well. Trained on random predators.
- Observe (visual) emergent behaviour by running the simulation
- Investigate real-life hunting scenarios by introducing reward shaping (ie: movement penalties, proximity reward, teamwork rewards)

World Focused (experimental block 3)
- The above, but with both predator and prey getting smarter - perhaps observe a nash-equilibrium as a convergence metric
- Perhaps continue training by using initialized trained weights 



# Initial Observations/Investigation Points:

- Predator reward is quite sparse and is distributed to every predator regardless
- Training is not going great with ~5000 episodes (1000 timesteps in each). Will need to change things up ?

# Things to change:
- Reward structure (may introduce bias)
- Predator/Prey numbering
- Timesteps in an episode
- FOV (both size and toggle), FOV off gives it 360º of view . 
- Speed
- Respawn vs non-respawning prey
- Catch radius (could fix the sparse reward density)
- Batch size?? (This could come into play if we use a GPU? Allowing us to run things in parrlel with eachther)
- Things to do with the ppo update itself ? (ie priorty, clipping??) Am I too conservative/aggressive?
- Add an initial "heuristic"



# Experiments:


1.1 -> densify the predator reward, try and get some sort of learning!
- 10+ prey and 1 predator, in a small tank. Ensure that the predator gets lucky
- PPO predator with parameter sharing versus random prey
- Keep all rewards baseline for now (ie no proximity, movement etc). Want to introduce no bias
- 5000 episodes (3 hour experiment.)
- No death of pred and respawning of prey (Should be better for getting some sort of signal)
- FOV true and catch radius 120, can tune these if we are seeing no signal
*** Perhaps we have to start off all training blocks like this??***
Results:
============================================================ 
        EXPERIMENT STARTED
        Run name       : ppo_vs_random_1774983186
        Predator agent : ppo  (obs=29, act=16)
        Prey agent     : random  (obs=29, act=16)
        Param sharing  : True
        Unique agents  : 2
        Episodes       : 5000
        Agents         : ['predator_0', 'prey_0', 'prey_1', 'prey_2', 'prey_3', 'prey_4', 'prey_5', 'prey_6', 'prey_7', 'prey_8', 'prey_9']
        max_steps      : 1000
        catch_radius   : 120
        fov_enabled    : True
        catch_bonus    : 10.0
        prox_reward    : 0.0
        prox_penalty   : 0.0
        step_penalty   : 0.0

        --> It died after I closed my laptop, not seeing much of a trend anyways.
        --> prey rewards getting fed into predator? (no)
        --> Try reducing the catch radius?
[Ep  3000/5000]  steps=1002   pred_r=   540.0  prey_r=-43980.0  pred0_loss=-0.0123  (0.0 ep/s, ETA 43482s)
[Ep  3050/5000]  steps=1002   pred_r=   510.0  prey_r=-40980.0  pred0_loss=-0.0063  (0.0 ep/s, ETA 42911s)
[Ep  3100/5000]  steps=1002   pred_r=   470.0  prey_r=-36980.0  pred0_loss=-0.0082  (0.0 ep/s, ETA 41550s)
[Ep  3150/5000]  steps=1002   pred_r=   530.0  prey_r=-42980.0  pred0_loss=-0.0041  (0.0 ep/s, ETA 41187s)
[Ep  3200/5000]  steps=1002   pred_r=   560.0  prey_r=-45980.0  pred0_loss=-0.0090  (0.0 ep/s, ETA 40329s)
[Ep  3250/5000]  steps=1002   pred_r=   560.0  prey_r=-45980.0  pred0_loss=-0.0063  (0.0 ep/s, ETA 39710s)
[Ep  3300/5000]  steps=1002   pred_r=   440.0  prey_r=-33980.0  pred0_loss=-0.0167  (0.0 ep/s, ETA 38187s)
[Ep  3350/5000]  steps=1002   pred_r=   510.0  prey_r=-40980.0  pred0_loss=-0.0080  (0.0 ep/s, ETA 36574s)
[Ep  3400/5000]  steps=1002   pred_r=   500.0  prey_r=-39980.0  pred0_loss=-0.0098  (0.0 ep/s, ETA 35004s)
[Ep  3450/5000]  steps=1002   pred_r=   460.0  prey_r=-35980.0  pred0_loss=-0.0119  (0.0 ep/s, ETA 33477s)
[Ep  3500/5000]  steps=1002   pred_r=   490.0  prey_r=-38980.0  pred0_loss=-0.0044  (0.0 ep/s, ETA 31989s)
[Ep  3550/5000]  steps=1002   pred_r=   440.0  prey_r=-33980.0  pred0_loss=-0.0078  (0.0 ep/s, ETA 30539s)
[Ep  3600/5000]  steps=1002   pred_r=   480.0  prey_r=-37980.0  pred0_loss=-0.0040  (0.0 ep/s, ETA 29126s)
============================================================
-> try to increase the tank size and decreased catch radius to make catching harder. Changed the reward to be more intuitive. decreased prey number and increased predator number. decreased episode steps (3000 episodes though). FOV off

    EXPERIMENT STARTED
    Run name       : ppo_vs_random_1775059543
    Predator agent : ppo  (obs=29, act=16)
    Prey agent     : random  (obs=29, act=16)
    Param sharing  : True
    Unique agents  : 2
    Episodes       : 3000
    Agents         : ['predator_0', 'prey_0', 'prey_1', 'prey_2', 'prey_3', 'prey_4']
    max_steps      : 690
    catch_radius   : 50
    fov_enabled    : False
    catch_bonus    : 10.0
    prox_reward    : 0.0
    prox_penalty   : 0.0
    step_penalty   : 0.0

    -> didnt really observe much learned behvaiour, exhibits a weird diagonal strategy though
    -> kinda just bumping into prey, want to create a scenario where it sees an advantage

============================================================
-> increased tank size (1200x1200), decreased # of prey to 3, increased episode length to 1000
  EXPERIMENT STARTED
  Run name       : ppo_vs_random_1775062767
  Predator agent : ppo  (obs=29, act=16)
  Prey agent     : random  (obs=29, act=16)
  Param sharing  : True
  Unique agents  : 2
  Episodes       : 3000
  Agents         : ['predator_0', 'prey_0', 'prey_1', 'prey_2']
  max_steps      : 1000
  catch_radius   : 50
  fov_enabled    : False
  catch_bonus    : 10.0
  prox_reward    : 0.0
  prox_penalty   : 0.0
  step_penalty   : 0.0

  --> looks more promising, need to keep tweaking the reward density though to encourage actual learning
  --> ran another with 1000x1000 tank size and 10,000 episodes. Rewards finally starting to happen. Going to increase it to 40,000 and run overnight
  --> Success at 40k! starting to see some real results
============================================================

Now we try it with MAPPO. 2 Runs, both 40,000 epds. One with starvation and another with no starvation
============================================================
  EXPERIMENT STARTED
  Run name       : mappo_vs_random_1775142219
  Predator agent : mappo  (obs=29, act=16)
  Prey agent     : random  (obs=29, act=16)
  Param sharing  : True
  Unique agents  : 2
  Episodes       : 40000
  Agents         : ['predator_0', 'prey_0', 'prey_1', 'prey_2']
  max_steps      : 1000
  catch_radius   : 50
  fov_enabled    : False
  catch_bonus    : 10.0
  prox_reward    : 0.0
  prox_penalty   : 0.0
  step_penalty   : 0.0
  no_catch_pen   : 0.0
========================================================

  --> Both PPO and MAPPO learn to catch at 40k, continued improvement at 100k
  --> No-catch penalty (-0.05) overwhelms catches — reward stays negative for both algos
  --> With 1 predator, MAPPO has no coordination advantage over PPO (centralized critic is wasted)



============================================================
  --> Now that we have been able to actually see rewards, lets move onto a 2-predator approach, since 1-predator 
  
  EXPERIMENT STARTED
  Run name       : mappo_vs_random_1776034272
  Predator agent : mappo  (obs=29, act=16)
  Prey agent     : random  (obs=29, act=16)
  Param sharing  : True
  Unique agents  : 2
  Episodes       : 40000
  Agents         : ['predator_0', 'predator_1', 'prey_0', 'prey_1', 'prey_2']
  max_steps      : 1000
  catch_radius   : 50
  fov_enabled    : False
  catch_bonus    : 10.0
  prox_reward    : 0.0
  prox_penalty   : 0.0
  step_penalty   : 0.0
  no_catch_pen   : 0.0
============================================================

============================================================
## Experiment Pipeline — Phases A/B/C
============================================================

### Summary of completed 1-predator runs (Block 1 baseline):
  - PPO  40k  baseline       → learning signal confirmed (results/ppo_1000_1k_3prey_40k)
  - PPO  40k  nocatchpen     → penalty too harsh, negative reward (results/ppo_1000_1k_3prey_40k_nocatchpen)
  - MAPPO 40k  baseline      → similar to PPO (results/mappo_1000_1k_3prey_40k)
  - MAPPO 40k  nocatchpen    → same pattern as PPO (results/mappo_1000_1k_3prey_40k_nocatchpen)
  - MAPPO 100k baseline      → continued improvement (results/mappo_1000_1k_3prey_100k)
  - PPO  100k baseline       → continued improvement (results/ppo_1000_1k_3prey_100k)

### Phase A — Multi-predator coordination (MAPPO should outperform PPO here)
  All: 1000x1000, 1000 steps, **2 predators** / 3 prey, catch 50, no FOV, v=2.5, no shaping, 100k

  A1: MAPPO 2-pred vs 3 random prey, 100k  → results/phase_a1_mappo_2pred_100k
  A2: PPO   2-pred vs 3 random prey, 100k  → results/phase_a2_ppo_2pred_100k

### Phase B — Prey training (Block 2)
  All: 1000x1000, 1000 steps, 1 random predator / **3 prey**, catch 50, no FOV, v=2.5, 100k

  B1: PPO   3-prey vs 1 random predator, 100k  → results/phase_b1_prey_ppo_100k
  B2: MAPPO 3-prey vs 1 random predator, 100k  → results/phase_b2_prey_mappo_100k

### Phase C — Co-evolution (Block 3)
  Load best weights from earlier phases, train the other side against them.

  C1: Train prey (MAPPO) against best predator weights, 100k
  C2: Train predator against best prey weights from C1, 100k

============================================================


