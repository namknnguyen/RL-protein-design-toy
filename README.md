# RL-protein-design-toy
A toy RL model for protein design, which I built to sketch out a rough pipeline for a future project. Doesn't do much for now, will be refined to actually work in a new project.

The model currently:

-Lets an RL agent modify a starting molecule by applying a set of pre-defined chemical transformations

-Computes loss by evaluating various physiochemical properties, as well as a Tanimoto similarity to a target molecule

-The idea is that we can train an agent to figure out a set of transformations that will create a small molecule with properties suitable for medicine and has great binding affinity to a target molecule by comparing its similarity to a known effective binder, and hopefully generalize to new starting molecules and target molecules. We can also expand this idea by creating models that figure out the shortest path to a target molecule which would benefit drug design.

Note that this model doesn't actually produce any results, only a rough proof-of-concept for me to visualize the pipeline for a future project and understand a bit more about reinforcement learning and molecule design.
