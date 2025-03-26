import os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

MAX_STEPS               = 30        # Number of transformations allowed per episode
NUM_EPISODES            = 5000      # Number of episodes
EPSILON                 = 0.1       # Exploration probability
BETA                    = 0.01      # Entropy bonus coefficient
LEARNING_RATE           = 5e-4      # Learning rate 
HIDDEN_DIM              = 256       # Hidden dim of policy network
STATE_DIM               = 6         # State vector dim
ACTION_DIM              = 18        # Number of actions
GAMMA                   = 0.99      # Discount factor for long-term rewards
TARGET_SCORE_MULTIPLIER = 4         # Weight for target binding score
MAX_ATTEMPTS            = 3         # Maximum attempts to apply a valid transformation
CHECKPOINT_FILE         = "checkpoint3.pth"  # Checkpoint filename

ALLOWED_VALENCES = {6: 4, 7: 3, 8: 2}

# Target SMILES for a common cancer immunotherapy target
DEFAULT_TARGET = "CN1CCN(CC1)C(=O)c1ccc(cc1)NC(=O)c1ccc(cc1)C"

def target_binding_predictor(smiles, target_smiles):
    mol = Chem.MolFromSmiles(smiles)
    target_mol = Chem.MolFromSmiles(target_smiles)
    if not mol or not target_mol:
        return -1.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2)
    similarity = DataStructs.TanimotoSimilarity(fp, target_fp)
    return similarity

Utility functions

def can_add_bond(rw_mol, atom_idx, additional=1):
    atom = rw_mol.GetAtomWithIdx(atom_idx)
    return atom.GetDegree() + additional <= ALLOWED_VALENCES.get(atom.GetAtomicNum(), 4)

def safe_sanitize(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                                  Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return mol, True
    except Exception:
        return mol, False

Transformations

def add_methyl(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    new_idx = rw.AddAtom(Chem.Atom('C'))
    rw.AddBond(idx, new_idx, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_oxygen(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    new_idx = rw.AddAtom(Chem.Atom('O'))
    rw.AddBond(idx, new_idx, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def remove_terminal_atom(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetDegree() == 1 and not a.IsInRing()]
    if not atoms:
        return mol
    idx = random.choice(atoms)
    try:
        rw.RemoveAtom(idx)
        new_mol, valid = safe_sanitize(rw.GetMol())
        return new_mol if valid else mol
    except Exception:
        return mol

def replace_carbon_with_nitrogen(mol):
    rw = Chem.RWMol(mol)
    atoms = [a for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    a = random.choice(atoms)
    if a.GetDegree() > ALLOWED_VALENCES.get(6, 4) - 1:
        return mol
    a.SetAtomicNum(7)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_carboxyl(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    carboxyl_idx = rw.AddAtom(Chem.Atom('C'))
    rw.AddBond(idx, carboxyl_idx, Chem.BondType.SINGLE)
    if not can_add_bond(rw, carboxyl_idx, additional=2):
        return mol
    oxy_db = rw.AddAtom(Chem.Atom('O'))
    rw.AddBond(carboxyl_idx, oxy_db, Chem.BondType.DOUBLE)
    oxy_sb = rw.AddAtom(Chem.Atom('O'))
    rw.AddBond(carboxyl_idx, oxy_sb, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_branch(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    branch_idx = rw.AddAtom(Chem.Atom('C'))
    rw.AddBond(idx, branch_idx, Chem.BondType.SINGLE)
    if not can_add_bond(rw, branch_idx):
        return mol
    oxy_idx = rw.AddAtom(Chem.Atom('O'))
    rw.AddBond(branch_idx, oxy_idx, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_phosphate(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    link_ox = rw.AddAtom(Chem.Atom('O'))
    P = rw.AddAtom(Chem.Atom('P'))
    rw.AddBond(link_ox, P, Chem.BondType.SINGLE)
    for _ in range(2):
        oxy = rw.AddAtom(Chem.Atom('O'))
        rw.AddBond(P, oxy, Chem.BondType.DOUBLE)
    oxy_sb = rw.AddAtom(Chem.Atom('O'))
    rw.AddBond(P, oxy_sb, Chem.BondType.SINGLE)
    H = rw.AddAtom(Chem.Atom('H'))
    rw.AddBond(oxy_sb, H, Chem.BondType.SINGLE)
    rw.AddBond(idx, link_ox, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_sulfonate(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    link_ox = rw.AddAtom(Chem.Atom('O'))
    S = rw.AddAtom(Chem.Atom('S'))
    rw.AddBond(link_ox, S, Chem.BondType.SINGLE)
    for _ in range(2):
        oxy = rw.AddAtom(Chem.Atom('O'))
        rw.AddBond(S, oxy, Chem.BondType.DOUBLE)
    oxy_sb = rw.AddAtom(Chem.Atom('O'))
    rw.AddBond(S, oxy_sb, Chem.BondType.SINGLE)
    H = rw.AddAtom(Chem.Atom('H'))
    rw.AddBond(oxy_sb, H, Chem.BondType.SINGLE)
    rw.AddBond(idx, link_ox, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_amine(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    N = rw.AddAtom(Chem.Atom('N'))
    rw.AddBond(idx, N, Chem.BondType.SINGLE)
    for _ in range(2):
        H = rw.AddAtom(Chem.Atom('H'))
        rw.AddBond(N, H, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_cyclopropyl(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    c1 = rw.AddAtom(Chem.Atom('C'))
    c2 = rw.AddAtom(Chem.Atom('C'))
    c3 = rw.AddAtom(Chem.Atom('C'))
    rw.AddBond(c1, c2, Chem.BondType.SINGLE)
    rw.AddBond(c2, c3, Chem.BondType.SINGLE)
    rw.AddBond(c3, c1, Chem.BondType.SINGLE)
    rw.AddBond(idx, c1, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_amide(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    amide_C = rw.AddAtom(Chem.Atom('C'))
    rw.AddBond(idx, amide_C, Chem.BondType.SINGLE)
    oxy = rw.AddAtom(Chem.Atom('O'))
    rw.AddBond(amide_C, oxy, Chem.BondType.DOUBLE)
    N = rw.AddAtom(Chem.Atom('N'))
    rw.AddBond(amide_C, N, Chem.BondType.SINGLE)
    for _ in range(2):
        H = rw.AddAtom(Chem.Atom('H'))
        rw.AddBond(N, H, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_phenyl(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    benzene = Chem.MolFromSmiles("c1ccccc1")
    mapping = {}
    for atom in benzene.GetAtoms():
        new_idx = rw.AddAtom(atom)
        mapping[atom.GetIdx()] = new_idx
    for bond in benzene.GetBonds():
        a1 = mapping[bond.GetBeginAtomIdx()]
        a2 = mapping[bond.GetEndAtomIdx()]
        rw.AddBond(a1, a2, bond.GetBondType())
    rw.AddBond(idx, mapping[0], Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_piperazine(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    piperazine = Chem.MolFromSmiles("N1CCNCC1")
    mapping = {}
    for atom in piperazine.GetAtoms():
        new_idx = rw.AddAtom(atom)
        mapping[atom.GetIdx()] = new_idx
    for bond in piperazine.GetBonds():
        a1 = mapping[bond.GetBeginAtomIdx()]
        a2 = mapping[bond.GetEndAtomIdx()]
        rw.AddBond(a1, a2, bond.GetBondType())
    rw.AddBond(idx, mapping[0], Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_fluorine(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    new_idx = rw.AddAtom(Chem.Atom('F'))
    rw.AddBond(idx, new_idx, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_methoxy(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    O_idx = rw.AddAtom(Chem.Atom('O'))
    CH3_idx = rw.AddAtom(Chem.Atom('C'))
    rw.AddBond(idx, O_idx, Chem.BondType.SINGLE)
    rw.AddBond(O_idx, CH3_idx, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_imidazole(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    imidazole = Chem.MolFromSmiles("c1cnc[nH]1")
    mapping = {}
    for atom in imidazole.GetAtoms():
        new_idx = rw.AddAtom(atom)
        mapping[atom.GetIdx()] = new_idx
    for bond in imidazole.GetBonds():
        a1 = mapping[bond.GetBeginAtomIdx()]
        a2 = mapping[bond.GetEndAtomIdx()]
        rw.AddBond(a1, a2, bond.GetBondType())
    rw.AddBond(idx, mapping[0], Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_pyridine(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    pyridine = Chem.MolFromSmiles("c1ccncc1")
    mapping = {}
    for atom in pyridine.GetAtoms():
        new_idx = rw.AddAtom(atom)
        mapping[atom.GetIdx()] = new_idx
    for bond in pyridine.GetBonds():
        a1 = mapping[bond.GetBeginAtomIdx()]
        a2 = mapping[bond.GetEndAtomIdx()]
        rw.AddBond(a1, a2, bond.GetBondType())
    rw.AddBond(idx, mapping[0], Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

def add_sulfonamide(mol):
    rw = Chem.RWMol(mol)
    atoms = [a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == 'C']
    if not atoms:
        return mol
    idx = random.choice(atoms)
    if not can_add_bond(rw, idx):
        return mol
    S_idx = rw.AddAtom(Chem.Atom('S'))
    O1_idx = rw.AddAtom(Chem.Atom('O'))
    O2_idx = rw.AddAtom(Chem.Atom('O'))
    N_idx = rw.AddAtom(Chem.Atom('N'))
    H1_idx = rw.AddAtom(Chem.Atom('H'))
    H2_idx = rw.AddAtom(Chem.Atom('H'))
    rw.AddBond(idx, S_idx, Chem.BondType.SINGLE)
    rw.AddBond(S_idx, O1_idx, Chem.BondType.DOUBLE)
    rw.AddBond(S_idx, O2_idx, Chem.BondType.DOUBLE)
    rw.AddBond(S_idx, N_idx, Chem.BondType.SINGLE)
    rw.AddBond(N_idx, H1_idx, Chem.BondType.SINGLE)
    rw.AddBond(N_idx, H2_idx, Chem.BondType.SINGLE)
    new_mol, valid = safe_sanitize(rw.GetMol())
    return new_mol if valid else mol

Transformation mapping
ACTIONS = {
    0: add_methyl, 1: add_oxygen, 2: remove_terminal_atom,
    3: replace_carbon_with_nitrogen, 4: add_carboxyl, 5: add_branch,
    6: add_phosphate, 7: add_sulfonate, 8: add_amine, 9: add_cyclopropyl,
    10: add_amide, 11: add_phenyl, 12: add_piperazine,
    13: add_fluorine, 14: add_methoxy, 15: add_imidazole, 16: add_pyridine, 17: add_sulfonamide
}

def apply_action(mol, action):
    return ACTIONS.get(action, lambda x: x)(mol)

 Reward functions
def compute_reward(mol, visited_fps, target_smiles):
    try:
        mw = Descriptors.MolWt(mol)
        qed_score = QED.qed(mol)
        logP = Descriptors.MolLogP(mol)
        target_score = target_binding_predictor(Chem.MolToSmiles(mol, canonical=True), target_smiles)
        rot = Descriptors.NumRotatableBonds(mol)
    except Exception:
        return -1.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    if not visited_fps:
        novelty = 1.0
        visited_fps.append(fp)
    else:
        sims = [DataStructs.TanimotoSimilarity(fp, v) for v in visited_fps]
        novelty = 1 - max(sims)
        if novelty > 0.3:
            visited_fps.append(fp)
    return (qed_score * 5 +
            target_score * TARGET_SCORE_MULTIPLIER -
            0.5 * abs(logP - 3) -
            max(0, (mw - 500) / 100) -
            rot * 0.1 +
            novelty * 2)

Environment
class MoleculeEnv:
    def __init__(self, max_steps=MAX_STEPS, target_smiles=DEFAULT_TARGET, starting_smiles='c1ccccc1'):
        self.max_steps = max_steps
        self.target_smiles = target_smiles  
        self.starting_smiles = starting_smiles
        self.reset()
    def reset(self):
        self.current_step = 0
        self.current_mol = Chem.MolFromSmiles(self.starting_smiles)
        self.visited_fps = []
        return Chem.MolToSmiles(self.current_mol, canonical=True)
    def step(self, action):
        self.current_step += 1
        new_mol = None
        for _ in range(MAX_ATTEMPTS):
            candidate = apply_action(self.current_mol, action)
            candidate, valid = safe_sanitize(candidate)
            if valid:
                new_mol = candidate
                break
        if new_mol is None:
            reward = -1.0
            smiles = Chem.MolToSmiles(self.current_mol, canonical=True)
        else:
            self.current_mol = new_mol
            smiles = Chem.MolToSmiles(self.current_mol, canonical=True)
            reward = compute_reward(self.current_mol, self.visited_fps, self.target_smiles)
        done = self.current_step >= self.max_steps
        return smiles, reward, done, {}

# RL Agent
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, output_dim=ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

class RLAgent:
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
                 lr=LEARNING_RATE, epsilon=EPSILON, beta=BETA):
        self.policy_net = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.epsilon = epsilon
        self.beta = beta
        self.action_dim = action_dim
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        probs = self.policy_net(state_tensor)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        if random.random() < self.epsilon:
            action = random.choice(range(self.action_dim))
            log_prob = torch.log(torch.tensor(1.0 / self.action_dim))
        else:
            action = np.random.choice(self.action_dim, p=probs.detach().numpy())
            log_prob = torch.log(probs[action] + 1e-8)
        return action, log_prob, entropy
    def update_policy(self, log_probs, rewards, entropies, gamma=GAMMA):
        discounted = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted.insert(0, R)
        discounted = torch.FloatTensor(discounted)
        loss = sum(-lp * R - self.beta * ent for lp, R, ent in zip(log_probs, discounted, entropies))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# Featurization
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [0] * STATE_DIM
    return [len(smiles), smiles.count('C'), smiles.count('O'),
            smiles.count('N'), QED.qed(mol), target_binding_predictor(smiles, DEFAULT_TARGET)]

# Checkpoints
def save_checkpoint(agent, filename=CHECKPOINT_FILE):
    torch.save({'model_state': agent.policy_net.state_dict(),
                'optimizer_state': agent.optimizer.state_dict()}, filename)
    print("Checkpoint saved.")
    
def load_checkpoint(agent, filename=CHECKPOINT_FILE):
    if os.path.exists(filename):
        ckpt = torch.load(filename)
        agent.policy_net.load_state_dict(ckpt['model_state'])
        agent.optimizer.load_state_dict(ckpt['optimizer_state'])
        print("Checkpoint loaded.")

# Train model
def train_agent(episodes=NUM_EPISODES, starting_smiles='c1ccccc1'):
    env = MoleculeEnv(max_steps=MAX_STEPS, target_smiles=DEFAULT_TARGET, starting_smiles=starting_smiles)
    agent = RLAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM,
                    lr=LEARNING_RATE, epsilon=EPSILON, beta=BETA)
    load_checkpoint(agent)
    
    best_reward = -float('inf')
    best_episode = None
    best_molecule = None
    
    for ep in range(episodes):
        state_smiles = env.reset()
        log_probs, rewards, entropies = [], [], []
        done = False
        while not done:
            state = featurize(state_smiles)
            action, lp, ent = agent.select_action(state)
            next_smiles, reward, done, _ = env.step(action)
            log_probs.append(lp)
            rewards.append(reward)
            entropies.append(ent)
            state_smiles = next_smiles
        cumulative_reward = sum(rewards)
        agent.update_policy(log_probs, rewards, entropies)
        
        if cumulative_reward > best_reward:
            best_reward = cumulative_reward
            best_episode = ep + 1
            best_molecule = state_smiles
            
        print(f"Ep {ep+1}: {state_smiles}, Cumulative Reward: {cumulative_reward:.2f}")
    
    save_checkpoint(agent)
    print("Training complete.")
    print(f"Best Episode: {best_episode}, Best Molecule: {best_molecule}, with Reward: {best_reward:.2f}")

if __name__ == '__main__':
    train_agent(starting_smiles='c1ccccc1')
