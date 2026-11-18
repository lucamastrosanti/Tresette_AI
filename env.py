import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from engine import Seme, Valore, Carta, Mazzo, Giocatore, Partita, DizionarioCarte

class BotEuristicoAvanzato:
    def __init__(self, env_ref):
        self.env = env_ref
        self.gerarchia_valori = ['QUATTRO', 'CINQUE', 'SEI', 'SETTE', 'FANTE', 'CAVALLO', 'RE', 'ASSO', 'DUE', 'TRE']
        self.punti_carta = {'ASSO': 3, 'DUE': 1, 'TRE': 1, 'RE': 1, 'CAVALLO': 1, 'FANTE': 1, 'SETTE': 0, 'SEI': 0, 'CINQUE': 0, 'QUATTRO': 0}

    def predict(self, obs, action_masks, deterministic=True):
        mosse_legali_idx = np.where(action_masks)[0]
        carte_legali = [self.env.dizionario.da_id_a_carta(idx) for idx in mosse_legali_idx]
        
        if len(carte_legali) == 1:
            return mosse_legali_idx[0], None

        mano_bot = self.env.avversario.mano
        carta_in_tavola = self.env.carta_in_tavola
        uscite = self.env.carte_uscite_ai + self.env.carte_uscite_avv

        if carta_in_tavola is None:
            dominanti = []
            for carta in mano_bot:
                idx_mia = self.gerarchia_valori.index(carta.valore.name)
                superiori_possibili = self.gerarchia_valori[idx_mia + 1:]
                è_dominante = True
                for sup_name in superiori_possibili:
                    carta_sup_str = f"{sup_name} di {carta.seme.name}"
                    if not any(str(c) == carta_sup_str for c in uscite) and not any(str(c) == carta_sup_str for c in mano_bot):
                        è_dominante = False
                        break
                if è_dominante: dominanti.append(carta)

            if dominanti:
                miglior_dominante = max(dominanti, key=lambda c: self.punti_carta[c.valore.name])
                return self.env.dizionario.da_carta_a_id(miglior_dominante), None

            conteggio_semi = {seme.name: 0 for seme in self.env.lista_semi}
            for c in mano_bot: conteggio_semi[c.seme.name] += 1
            seme_lungo_name = max(conteggio_semi, key=conteggio_semi.get)
            carte_seme_lungo = [c for c in mano_bot if c.seme.name == seme_lungo_name]
            
            peggior_carta = min(carte_seme_lungo, key=lambda c: self.gerarchia_valori.index(c.valore.name))
            return self.env.dizionario.da_carta_a_id(peggior_carta), None

        else:
            seme_richiesto = carta_in_tavola.seme
            if carte_legali[0].seme != seme_richiesto:
                scarto = min(carte_legali, key=lambda c: (self.punti_carta[c.valore.name], self.gerarchia_valori.index(c.valore.name)))
                return self.env.dizionario.da_carta_a_id(scarto), None

            forza_tavola = self.gerarchia_valori.index(carta_in_tavola.valore.name)
            carte_vincenti = [c for c in carte_legali if self.gerarchia_valori.index(c.valore.name) > forza_tavola]

            if carte_vincenti:
                if self.punti_carta[carta_in_tavola.valore.name] >= 1:
                    miglior_vincente = max(carte_vincenti, key=lambda c: self.gerarchia_valori.index(c.valore.name))
                else:
                    miglior_vincente = min(carte_vincenti, key=lambda c: self.gerarchia_valori.index(c.valore.name))
                return self.env.dizionario.da_carta_a_id(miglior_vincente), None
            else:
                scarto = min(carte_legali, key=lambda c: (self.punti_carta[c.valore.name], self.gerarchia_valori.index(c.valore.name)))
                return self.env.dizionario.da_carta_a_id(scarto), None

class TressetteEnv(gym.Env):
    def __init__(self, fase_curriculum="completa"):
        super(TressetteEnv, self).__init__()
        self.dizionario = DizionarioCarte()
        self.action_space = spaces.Discrete(40)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(132,), dtype=np.float32)
        self.lista_semi = list(Seme)
        self.pool_avversari = []
        self.modelli_cache = {}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.ai = Giocatore("Agente_AI")
        self.avversario = Giocatore("Bot_Avversario")
        self.arbitro = Partita(self.ai, self.avversario)
        
        if len(self.pool_avversari) > 0:
            if random.random() < 0.8:
                nome_scelto = random.choice(self.pool_avversari)
                if nome_scelto not in self.modelli_cache:
                    from sb3_contrib import MaskablePPO
                    self.modelli_cache[nome_scelto] = MaskablePPO.load(nome_scelto)
                self.modello_avversario = self.modelli_cache[nome_scelto]
            else:
                self.modello_avversario = BotEuristicoAvanzato(self)
        else:
            self.modello_avversario = BotEuristicoAvanzato(self)

        self.arbitro.dai_carte()
                
        self.turno_attuale = 1 
        self.carte_uscite_ai = []
        self.carte_uscite_avv = []
            
        self.ai.ordina_mano()
        self.avversario.ordina_mano()
        self.ai.punti_accuso = 0
        self.avversario.punti_accuso = 0
        self.piombi_ai = {seme: 0.0 for seme in Seme}
        self.piombi_avv = {seme: 0.0 for seme in Seme}
        self.carta_in_tavola = None
        self.ultimo_vincitore = None
        
        self.primo = random.choice([self.ai, self.avversario])
        self.secondo = self.avversario if self.primo == self.ai else self.ai
        
        if self.primo == self.avversario:
            carta_scelta = self._ottieni_mossa_avversario()
            self.carta_in_tavola = self.avversario.gioca_carta(carta_scelta)
            self.carte_uscite_avv.append(self.carta_in_tavola)
            
        return self._get_obs(), {}

    def aggiorna_pool(self, nuova_pool):
        self.pool_avversari = nuova_pool

    def _get_obs(self, prospettiva_avversario=False):
        mio_gioc = self.avversario if prospettiva_avversario else self.ai
        suo_gioc = self.ai if prospettiva_avversario else self.avversario
        miei_piombi = self.piombi_avv if prospettiva_avversario else self.piombi_ai
        suoi_piombi = self.piombi_ai if prospettiva_avversario else self.piombi_avv

        mano_vec = np.zeros(40, dtype=np.float32)
        for c in mio_gioc.mano: mano_vec[self.dizionario.da_carta_a_id(c)] = 1.0
            
        tavolo_vec = np.zeros(40, dtype=np.float32)
        if self.carta_in_tavola is not None:
            tavolo_vec[self.dizionario.da_carta_a_id(self.carta_in_tavola)] = 1.0

        uscite_vec = np.zeros(40, dtype=np.float32)
        for c in mio_gioc.prese: uscite_vec[self.dizionario.da_carta_a_id(c)] = 1.0
        for c in suo_gioc.prese: uscite_vec[self.dizionario.da_carta_a_id(c)] = -1.0

        piombi_miei_vec = np.array([miei_piombi[s] for s in self.lista_semi], dtype=np.float32)
        piombi_suoi_vec = np.array([suoi_piombi[s] for s in self.lista_semi], dtype=np.float32)

        turno_norm = self.turno_attuale / 20.0
        sono_primo = 1.0 if self.primo == mio_gioc else 0.0
        punti_mio = min(mio_gioc.calcola_punti_totali() / 11.0, 1.0)
        punti_suo = min(suo_gioc.calcola_punti_totali() / 11.0, 1.0)
        extra_vec = np.array([turno_norm, sono_primo, punti_mio, punti_suo], dtype=np.float32)

        return np.concatenate((mano_vec, tavolo_vec, uscite_vec, piombi_miei_vec, piombi_suoi_vec, extra_vec))

    def _ottieni_mossa_avversario(self):
        seme_richiesto = self.carta_in_tavola.seme if self.carta_in_tavola else None
        
        obs = self._get_obs(prospettiva_avversario=True)
        maschera = self._get_maschera_avversario()
        azione, _ = self.modello_avversario.predict(obs, action_masks=maschera, deterministic=False)
        carta_teorica = self.dizionario.da_id_a_carta(int(azione))
        return next(c for c in self.avversario.mano if str(c) == str(carta_teorica))

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        carta_giocata_ai = self.dizionario.da_id_a_carta(action)
        ha_la_carta = any(str(c) == str(carta_giocata_ai) for c in self.ai.mano)
        seme_richiesto = self.carta_in_tavola.seme if self.carta_in_tavola else None
        mossa_legale = self.arbitro.mossa_valida(carta_giocata_ai, self.ai.mano, seme_richiesto) if ha_la_carta else False

        if not ha_la_carta or not mossa_legale:
            reward = -100.0
            terminated = True
            info["motivo"] = "Mossa Illegale"
            return self._get_obs(), reward, terminated, truncated, info

        carta_reale = next(c for c in self.ai.mano if str(c) == str(carta_giocata_ai))
        self.ai.gioca_carta(carta_reale)
        self.carte_uscite_ai.append(carta_reale)

        if self.carta_in_tavola is not None and carta_reale.seme != seme_richiesto:
            self.piombi_ai[seme_richiesto] = 1.0

        if self.primo == self.ai:
            self.carta_in_tavola = carta_reale
            carta_avversario = self._ottieni_mossa_avversario()
            self.avversario.gioca_carta(carta_avversario)
            self.carte_uscite_avv.append(carta_avversario)
            if carta_avversario.seme != self.carta_in_tavola.seme:
                self.piombi_avv[self.carta_in_tavola.seme] = 1.0
            carta1, carta2 = carta_reale, carta_avversario
            primo_gioc, secondo_gioc = self.ai, self.avversario
        else:
            carta1, carta2 = self.carta_in_tavola, carta_reale
            primo_gioc, secondo_gioc = self.avversario, self.ai

        vincitore_presa, carte_vinte = self.arbitro.valuta_presa(carta1, carta2, primo_gioc, secondo_gioc)
        vincitore_presa.prese.extend(carte_vinte)
        self.ultimo_vincitore = vincitore_presa
        
        if vincitore_presa == self.ai: self.primo = self.ai
        else: self.primo = self.avversario

        if len(self.arbitro.mazzo.carte) > 0:
            perdente = self.avversario if vincitore_presa == self.ai else self.ai
            vincitore_presa.mano.append(self.arbitro.mazzo.pesca(1)[0])
            perdente.mano.append(self.arbitro.mazzo.pesca(1)[0])
            self.ai.ordina_mano()
            self.avversario.ordina_mano()
            self.piombi_ai = {seme: 0.0 for seme in Seme}
            self.piombi_avv = {seme: 0.0 for seme in Seme}

        self.carta_in_tavola = None
        self.turno_attuale += 1

        if self.turno_attuale > 20:
            terminated = True
            punti_puri_ai = sum(c.punteggio for c in self.ai.prese) / 3.0
            punti_puri_avv = sum(c.punteggio for c in self.avversario.prese) / 3.0
            
            if self.ultimo_vincitore == self.ai: punti_puri_ai += 1
            else: punti_puri_avv += 1
                
            reward = punti_puri_ai - punti_puri_avv
            if punti_puri_ai > punti_puri_avv: reward += 5.0
            elif punti_puri_avv > punti_puri_ai: reward -= 5.0
        else:
            if self.primo == self.avversario:
                carta_scelta = self._ottieni_mossa_avversario()
                self.carta_in_tavola = self.avversario.gioca_carta(carta_scelta)
                self.carte_uscite_avv.append(self.carta_in_tavola)
                info["carta_giocata_ai"] = self.carta_in_tavola

        return self._get_obs(), reward, terminated, truncated, info
    
    def action_masks(self):
        maschera = np.zeros(40, dtype=bool)
        seme_richiesto = self.carta_in_tavola.seme if self.carta_in_tavola else None
        for c in self.ai.mano:
            if self.arbitro.mossa_valida(c, self.ai.mano, seme_richiesto):
                maschera[self.dizionario.da_carta_a_id(c)] = True
        return maschera
        
    def _get_maschera_avversario(self):
        maschera = np.zeros(40, dtype=bool)
        seme_richiesto = self.carta_in_tavola.seme if self.carta_in_tavola else None
        for c in self.avversario.mano:
            if self.arbitro.mossa_valida(c, self.avversario.mano, seme_richiesto):
                maschera[self.dizionario.da_carta_a_id(c)] = True
        return maschera

    def imposta_modello_avversario(self, modello):
        self.modello_avversario = modello