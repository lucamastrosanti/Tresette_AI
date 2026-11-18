import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from env import TressetteEnv
from sb3_contrib import MaskablePPO
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import uuid
from supabase import create_client, Client
import sys

torch.set_num_threads(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = "https://shfjmqoilrrelrldwqla.supabase.co"
SUPABASE_KEY = "sb_publishable_0ROKDnjFOiM4xr9hIrwnvA_n_IwKDkE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

modello_ai = None
partite_attive = {}

class MossaGiocatore(BaseModel):
    id_carta: int
    session_id: str

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

setattr(sys.modules['__main__'], 'TressetteFeaturesExtractor', TressetteFeaturesExtractor)

class BotIbrido:
    def __init__(self, modello_base, env_ref, num_mondi=20):
        self.modello = modello_base
        self.env_ref = env_ref
        self.num_mondi = num_mondi
        self.in_simulazione = False

    def predict(self, obs, action_masks, deterministic=True):
        if self.in_simulazione:
            return self.modello.predict(obs, action_masks=action_masks, deterministic=True)

        mosse_legali = np.where(action_masks)[0]
        if len(mosse_legali) == 1:
            return mosse_legali[0], None

        if len(self.env_ref.arbitro.mazzo.carte) == 0:
            return self._calcola_mossa_perfetta(mosse_legali), None

        valutazioni_mosse = {mossa: 0.0 for mossa in mosse_legali}
        
        carte_bot = [self.env_ref.dizionario.da_carta_a_id(c) for c in self.env_ref.avversario.mano]
        uscite = [self.env_ref.dizionario.da_carta_a_id(c) for c in self.env_ref.carte_uscite_ai + self.env_ref.carte_uscite_avv]
        in_tavola = [self.env_ref.dizionario.da_carta_a_id(self.env_ref.carta_in_tavola)] if self.env_ref.carta_in_tavola else []
        
        carte_note = set(carte_bot + uscite + in_tavola)
        carte_ignote = [i for i in range(40) if i not in carte_note]
        num_umano = len(self.env_ref.ai.mano)

        self.in_simulazione = True

        for _ in range(self.num_mondi):
            random.shuffle(carte_ignote)
            mano_umano = carte_ignote[:num_umano]

            for mossa in mosse_legali:
                modello_temp = self.env_ref.modello_avversario
                self.env_ref.modello_avversario = None
                
                env_sim = copy.deepcopy(self.env_ref)
                self.env_ref.modello_avversario = modello_temp
                
                env_sim.ai.mano = [env_sim.dizionario.da_id_a_carta(i) for i in mano_umano]
                carta_mossa = env_sim.dizionario.da_id_a_carta(mossa)
                
                if env_sim.carta_in_tavola is None: 
                    env_sim.carta_in_tavola = env_sim.avversario.gioca_carta(carta_mossa)
                    env_sim.carte_uscite_avv.append(env_sim.carta_in_tavola)
                    
                    obs_umano = env_sim._get_obs(prospettiva_avversario=False)
                    maschera_umano = env_sim.action_masks()
                    azione_umano, _ = self.modello.predict(obs_umano, action_masks=maschera_umano, deterministic=True)
                    carta_umano = env_sim.dizionario.da_id_a_carta(int(azione_umano))
                    env_sim.ai.gioca_carta(carta_umano)
                    env_sim.carte_uscite_ai.append(carta_umano)
                    
                    vincitore, prese = env_sim.arbitro.valuta_presa(env_sim.carta_in_tavola, carta_umano, env_sim.avversario, env_sim.ai)
                else: 
                    carta_umano = env_sim.carta_in_tavola
                    env_sim.avversario.gioca_carta(carta_mossa)
                    env_sim.carte_uscite_avv.append(carta_mossa)
                    vincitore, prese = env_sim.arbitro.valuta_presa(carta_umano, carta_mossa, env_sim.ai, env_sim.avversario)
                    
                obs_futuro = env_sim._get_obs(prospettiva_avversario=True)
                obs_tensor = torch.tensor(obs_futuro, dtype=torch.float32).unsqueeze(0).to(self.modello.device)
                
                with torch.no_grad():
                    valore_futuro_atteso = self.modello.policy.predict_values(obs_tensor).item()
                    
                se_prende_ora = 1.0 if vincitore == env_sim.avversario else -1.0
                valutazioni_mosse[mossa] += (valore_futuro_atteso + se_prende_ora)

        self.in_simulazione = False
        miglior_mossa = max(valutazioni_mosse, key=valutazioni_mosse.get)
        return miglior_mossa, None

    def _calcola_mossa_perfetta(self, mosse_legali_idx):
        tutte_le_carte = [self.env_ref.dizionario.da_id_a_carta(i) for i in range(40)]
        carte_bot = self.env_ref.avversario.mano
        uscite = self.env_ref.carte_uscite_ai + self.env_ref.carte_uscite_avv
        in_tavola = [self.env_ref.carta_in_tavola] if self.env_ref.carta_in_tavola else []
        
        carte_viste_nomi = set(str(c) for c in carte_bot + uscite + in_tavola)
        mano_umano = [c for c in tutte_le_carte if str(c) not in carte_viste_nomi]

        miglior_punteggio = -1000
        miglior_mossa_id = mosse_legali_idx[0]
        memo_cache = {}

        for mossa_id in mosse_legali_idx:
            carta_mossa = self.env_ref.dizionario.da_id_a_carta(mossa_id)
            nuova_mano_bot = [c for c in carte_bot if str(c) != str(carta_mossa)]
            
            if self.env_ref.carta_in_tavola is None:
                punti = self._minimax(nuova_mano_bot, mano_umano, carta_mossa, bot_to_play=False, memo=memo_cache)
            else:
                bot_wins, p_presi = self._risolvi_presa(self.env_ref.carta_in_tavola, carta_mossa, leader_is_bot=False)
                punti = p_presi + self._minimax(nuova_mano_bot, mano_umano, None, bot_to_play=bot_wins, memo=memo_cache)
            
            if punti > miglior_punteggio:
                miglior_punteggio = punti
                miglior_mossa_id = mossa_id

        return miglior_mossa_id

    def _minimax(self, mano_bot, mano_umano, carta_in_tavola, bot_to_play, alpha=-1000, beta=1000, memo=None):
        if not mano_bot and not mano_umano:
            return 0
            
        chiave = (
            frozenset(str(c) for c in mano_bot), 
            frozenset(str(c) for c in mano_umano), 
            str(carta_in_tavola) if carta_in_tavola else None, 
            bot_to_play
        )
        if chiave in memo: return memo[chiave]

        mano_corrente = mano_bot if bot_to_play else mano_umano
        
        seme_req = carta_in_tavola.seme if carta_in_tavola else None
        mosse_possibili = [c for c in mano_corrente if self.env_ref.arbitro.mossa_valida(c, mano_corrente, seme_req)]

        if bot_to_play:
            max_eval = -1000
            for mossa in mosse_possibili:
                nuova_mano = [c for c in mano_bot if str(c) != str(mossa)]
                if carta_in_tavola is None:
                    punti = self._minimax(nuova_mano, mano_umano, mossa, False, alpha, beta, memo)
                else:
                    bot_wins, p_presi = self._risolvi_presa(carta_in_tavola, mossa, leader_is_bot=False)
                    punti = p_presi + self._minimax(nuova_mano, mano_umano, None, bot_wins, alpha, beta, memo)
                
                max_eval = max(max_eval, punti)
                alpha = max(alpha, punti)
                if beta <= alpha: break
            memo[chiave] = max_eval
            return max_eval
        else:
            min_eval = 1000
            for mossa in mosse_possibili:
                nuova_mano = [c for c in mano_umano if str(c) != str(mossa)]
                if carta_in_tavola is None:
                    punti = self._minimax(mano_bot, nuova_mano, mossa, True, alpha, beta, memo)
                else:
                    bot_wins, p_presi = self._risolvi_presa(carta_in_tavola, mossa, leader_is_bot=True)
                    punti = p_presi + self._minimax(mano_bot, nuova_mano, None, bot_wins, alpha, beta, memo)
                
                min_eval = min(min_eval, punti)
                beta = min(beta, punti)
                if beta <= alpha: break
            memo[chiave] = min_eval
            return min_eval

    def _risolvi_presa(self, carta1, carta2, leader_is_bot):
        punti_tot = carta1.punteggio + carta2.punteggio
        if carta2.seme == carta1.seme and carta2.forza > carta1.forza:
            bot_wins = not leader_is_bot
        else:
            bot_wins = leader_is_bot
        return bot_wins, (punti_tot if bot_wins else 0)

def aggiorna_log_statistiche(punti_puri_umano, punti_puri_bot):
    try:
        dati_partita = {
            "punti_umano": float(punti_puri_umano),
            "punti_bot": float(punti_puri_bot)
        }
        supabase.table("partite_tressette").insert(dati_partita).execute()
        print(f"\n📊 PARTITA SALVATA SU SUPABASE! Bot: {punti_puri_bot:.2f} | Tu: {punti_puri_umano:.2f}\n")
    except Exception as e:
        print(f"\n⚠️ Errore di salvataggio su Supabase: {e}\n")

@app.on_event("startup")
def carica_modello():
    global modello_ai
    print("🤖 Caricamento del super-cervello strategico AI in corso...")
    
    ambiente_base = TressetteEnv()
    
    oggetti_custom = {
        "policy_kwargs": {
            "features_extractor_class": TressetteFeaturesExtractor,
            "features_extractor_kwargs": {"features_dim": 128},
            "net_arch": [512, 512]
        }
    }
    
    modello_ai = MaskablePPO.load(
        "tressette_bot_supremo_assoluto", 
        env=ambiente_base, 
        custom_objects=oggetti_custom,
        device="cpu"
    ) 
    print("✅ Modello caricato con successo!")

@app.post("/api/nuova_partita")
def nuova_partita():
    global partite_attive, modello_ai
    
    session_id = uuid.uuid4().hex
    ambiente_gioco = TressetteEnv()
    
    bot_potenziato = BotIbrido(modello_base=modello_ai, env_ref=ambiente_gioco, num_mondi=20)
    ambiente_gioco.imposta_modello_avversario(bot_potenziato)
    
    ambiente_gioco.reset()
    partite_attive[session_id] = ambiente_gioco
    
    mano_umano = [{"id": ambiente_gioco.dizionario.da_carta_a_id(c), "nome": str(c)} for c in ambiente_gioco.ai.mano]
    carta_in_tavola_dict = None
    if ambiente_gioco.carta_in_tavola:
        carta_in_tavola_dict = {
            "id": ambiente_gioco.dizionario.da_carta_a_id(ambiente_gioco.carta_in_tavola),
            "nome": str(ambiente_gioco.carta_in_tavola)
        }
        
    return {
        "session_id": session_id,
        "messaggio": "Nuova partita iniziata",
        "mano": mano_umano,
        "carta_in_tavola": carta_in_tavola_dict
    }

@app.post("/api/gioca_carta")
def gioca_carta(mossa: MossaGiocatore):
    global partite_attive
    
    if mossa.session_id not in partite_attive:
        raise HTTPException(status_code=400, detail="Partita inesistente o scaduta. Riavvia la pagina.")
        
    ambiente_gioco = partite_attive[mossa.session_id]
        
    maschera_azioni = ambiente_gioco.action_masks()
    if not maschera_azioni[mossa.id_carta]:
        return {"errore": True, "messaggio": "Mossa illegale! Devi rispondere a seme."}

    umano_inizia = (ambiente_gioco.carta_in_tavola is None)
    mano_ai_prima = set(ambiente_gioco.avversario.mano)
        
    obs, reward, done, truncated, info = ambiente_gioco.step(mossa.id_carta)
    
    mano_ai_dopo = set(ambiente_gioco.avversario.mano)
    carte_perse_da_ai = list(mano_ai_prima - mano_ai_dopo)
    carta_giocata_da_ai = None

    if not umano_inizia:
        carta_giocata_da_ai = None
    else:
        if len(carte_perse_da_ai) == 1:
            bot_carta = carte_perse_da_ai[0]
        elif len(carte_perse_da_ai) > 1:
            if ambiente_gioco.carta_in_tavola in carte_perse_da_ai:
                carte_perse_da_ai.remove(ambiente_gioco.carta_in_tavola)
            bot_carta = carte_perse_da_ai[0] if carte_perse_da_ai else None
        else:
            bot_carta = None

        if bot_carta:
            carta_giocata_da_ai = {
                "id": ambiente_gioco.dizionario.da_carta_a_id(bot_carta),
                "nome": str(bot_carta)
            }

    if done:
        punti_puri_bot = sum(c.punteggio for c in ambiente_gioco.avversario.prese) / 3.0
        punti_puri_umano = sum(c.punteggio for c in ambiente_gioco.ai.prese) / 3.0
        
        if ambiente_gioco.ultimo_vincitore == ambiente_gioco.avversario:
            punti_puri_bot += 1.0
        elif ambiente_gioco.ultimo_vincitore == ambiente_gioco.ai:
            punti_puri_umano += 1.0
            
        aggiorna_log_statistiche(punti_puri_umano, punti_puri_bot)
        del partite_attive[mossa.session_id]

    mano_aggiornata = [{"id": ambiente_gioco.dizionario.da_carta_a_id(c), "nome": str(c)} for c in ambiente_gioco.ai.mano]
    
    nuova_carta_in_tavola = None
    if ambiente_gioco.carta_in_tavola:
        nuova_carta_in_tavola = {
            "id": ambiente_gioco.dizionario.da_carta_a_id(ambiente_gioco.carta_in_tavola),
            "nome": str(ambiente_gioco.carta_in_tavola)
        }

    return {
        "errore": False,
        "finito": done,
        "mano": mano_aggiornata,
        "carta_giocata_da_ai": carta_giocata_da_ai,
        "nuova_carta_in_tavola": nuova_carta_in_tavola,
        "ultimo_vincitore": str(ambiente_gioco.ultimo_vincitore.nome) if ambiente_gioco.ultimo_vincitore else None,
        "punti_umano": ambiente_gioco.ai.calcola_punti_totali() if not done else punti_puri_umano,
        "punti_ai": ambiente_gioco.avversario.calcola_punti_totali() if not done else punti_puri_bot
    }