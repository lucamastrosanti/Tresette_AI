from enum import Enum
import random

class Seme(Enum):
    DENARI = 'denari'
    COPPE = 'coppe'
    SPADE = 'spade'
    BASTONI = 'bastoni'

class Valore(Enum):
    TRE = 3
    DUE = 2
    ASSO = 1
    RE = 10
    CAVALLO = 9
    FANTE = 8
    SETTE = 7
    SEI = 6
    CINQUE = 5
    QUATTRO = 4

class Carta:
    def __init__(self, valore: Valore, seme: Seme):
        self.valore = valore
        self.seme = seme
        self.forza = self._calcola_forza()
        self.punteggio = self._calcola_punteggio()

    def _calcola_forza(self):
        gerarchia = {
            Valore.TRE: 10,
            Valore.DUE: 9,
            Valore.ASSO: 8,
            Valore.RE: 7,
            Valore.CAVALLO: 6,
            Valore.FANTE: 5,
            Valore.SETTE: 4,
            Valore.SEI: 3,
            Valore.CINQUE: 2,
            Valore.QUATTRO: 1
        }
        return gerarchia[self.valore]

    def _calcola_punteggio(self):
        if self.valore == Valore.ASSO:
            return 3
        elif self.valore in [Valore.DUE, Valore.TRE, Valore.RE, Valore.CAVALLO, Valore.FANTE]:
            return 1
        else:
            return 0

    def __str__(self):
        return f"{self.valore.name} di {self.seme.name}"
    
    def __repr__(self):
        return self.__str__()
    

class Mazzo:
    def __init__(self):
        self.carte = []
        self._crea_mazzo()

    def _crea_mazzo(self):
        for seme in Seme:
            for valore in Valore:
                self.carte.append(Carta(valore, seme))

    def mescola(self):
        random.shuffle(self.carte)

    def pesca(self, numero_carte=1):
        pescate = []
        for _ in range(numero_carte):
            if self.carte:
                pescate.append(self.carte.pop(0))
        return pescate
    


class Giocatore:
    def __init__(self, nome: str):
        self.nome = nome
        self.mano = []
        self.prese = []
        self.punti_accuso = 0

    def gioca_carta(self, carta: Carta):
        self.mano.remove(carta)
        return carta

    def ordina_mano(self):
        self.mano.sort(key=lambda c: (c.seme.name, c.forza))

    def calcola_accusi(self):
        self.punti_accuso = 0
        for seme in Seme:
            ha_tre = any(c.valore == Valore.TRE and c.seme == seme for c in self.mano)
            ha_due = any(c.valore == Valore.DUE and c.seme == seme for c in self.mano)
            ha_asso = any(c.valore == Valore.ASSO and c.seme == seme for c in self.mano)
            
            if ha_tre and ha_due and ha_asso:
                self.punti_accuso += 3

        for valore_cercato in [Valore.TRE, Valore.DUE, Valore.ASSO]:
            conteggio = sum(1 for c in self.mano if c.valore == valore_cercato)
            if conteggio == 3:
                self.punti_accuso += 3
            elif conteggio == 4:
                self.punti_accuso += 4

    def calcola_punti_totali(self):
        totale_terzi = sum(carta.punteggio for carta in self.prese)
        return totale_terzi / 3 + self.punti_accuso

    def __str__(self):
        return self.nome
    

class Partita:
    def __init__(self, giocatore1: Giocatore, giocatore2: Giocatore):
        self.g1 = giocatore1
        self.g2 = giocatore2
        self.mazzo = Mazzo()
        self.mazzo.mescola()

    def dai_carte(self):
        self.g1.mano = self.mazzo.pesca(10)
        self.g2.mano = self.mazzo.pesca(10)

    def mossa_valida(self, carta_scelta: Carta, mano_giocatore: list, seme_di_partenza: Seme = None):
        if seme_di_partenza is None:
            return True
            
        ha_il_seme = any(c.seme == seme_di_partenza for c in mano_giocatore)
        
        if ha_il_seme and carta_scelta.seme != seme_di_partenza:
            return False
            
        return True

    def valuta_presa(self, carta1: Carta, carta2: Carta, primo_gioc: Giocatore, secondo_gioc: Giocatore):
        seme_partenza = carta1.seme
        
        if carta2.seme == seme_partenza and carta2.forza > carta1.forza:
            vincitore = secondo_gioc
        else:
            vincitore = primo_gioc

        carte_in_tavola = [carta1, carta2]
        return vincitore, carte_in_tavola
    

class DizionarioCarte:
    def __init__(self):
        self.id_a_carta = {}
        self.carta_a_id = {}
        
        indice = 0
        for seme in Seme:
            for valore in Valore:
                carta_fissa = Carta(valore, seme)
                self.id_a_carta[indice] = carta_fissa
                self.carta_a_id[str(carta_fissa)] = indice 
                indice += 1

    def da_id_a_carta(self, id_carta: int) -> Carta:
        return self.id_a_carta[id_carta]

    def da_carta_a_id(self, carta: Carta) -> int:
        return self.carta_a_id[str(carta)]

