import time
import torch
import torch.nn as nn
from typing import Callable
from env import TressetteEnv
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class TressetteFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        self.mano_net = nn.Sequential(nn.Linear(40, 32), nn.ReLU())
        self.tavolo_net = nn.Sequential(nn.Linear(40, 32), nn.ReLU())
        self.uscite_net = nn.Sequential(nn.Linear(40, 32), nn.ReLU())
        self.contesto_net = nn.Sequential(nn.Linear(12, 32), nn.ReLU())

    def forward(self, observations):
        mano = observations[:, 0:40]
        tavolo = observations[:, 40:80]
        uscite = observations[:, 80:120]
        contesto = observations[:, 120:132]

        val_mano = self.mano_net(mano)
        val_tavolo = self.tavolo_net(tavolo)
        val_uscite = self.uscite_net(uscite)
        val_contesto = self.contesto_net(contesto)

        return torch.cat((val_mano, val_tavolo, val_uscite, val_contesto), dim=1)

def crea_env_completo():
    return Monitor(TressetteEnv())

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

if __name__ == '__main__':
    print("=== INIZIO EVOLUZIONE SUPREMA (SELF-PLAY + MEMORIA + ESPERTO) ===")
    start_time = time.time()
    
    num_cpu = 8 
    env_completo = SubprocVecEnv([crea_env_completo for _ in range(num_cpu)])

    print("Caricamento del modello...")
    modello_campione = MaskablePPO.load(
        "tressette_bot_definitivo_vs_esperto",
        env=env_completo,
        custom_objects={
            "observation_space": env_completo.observation_space,
            "learning_rate": linear_schedule(5e-5), 
            "clip_range": 0.1, 
            "ent_coef": 0.005 
        }
    )
    
    modello_campione.tensorboard_log = "./logs_evoluzione_suprema/"
    
    generazioni_complete = 20
    step_per_gen_completo = 1_000_000
    pool_avversari = []

    for gen in range(1, generazioni_complete + 1):
        print(f"\n[TRAINING] GENERAZIONE {gen}/{generazioni_complete}")
        modello_campione.learn(total_timesteps=step_per_gen_completo, reset_num_timesteps=False)
        
        nome_modello_finale = f"bot_supremo_gen_{gen}"
        modello_campione.save(nome_modello_finale)
        
        pool_avversari.append(nome_modello_finale)
        if len(pool_avversari) > 10:
            pool_avversari.pop(0)
            
        env_completo.env_method("aggiorna_pool", pool_avversari)

    modello_campione.save("tressette_bot_supremo_assoluto")
    env_completo.close()
    
    tempo_totale = round((time.time() - start_time) / 3600, 2)
    print(f"\n=== TRAINING COMPLETATO IN {tempo_totale} ORE ===")