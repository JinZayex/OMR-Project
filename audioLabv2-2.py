#!/usr/bin/env python3
"""
Grid search per ottimizzazione iperparametri YOLO
CON SALVATAGGIO INCREMENTALE - Monitorabile in tempo reale
"""

from ultralytics import YOLO
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

# ===== CONFIGURAZIONE =====
DATA_PATH = "/Users/antonio/Desktop/AudioLabs_v2_DEST/data.yaml"
SAVE_DIR = Path("/Users/antonio/runs/grid_search")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BASE_PARAMS = {
    "batch": 8,
    "workers": 4,
    "imgsz": 640,
    "epochs": 20,  # Ridotte per velocità
    "fliplr": 0.8,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.3,
    "optimizer": "AdamW",
    "patience": 10,
}

# Spazio di ricerca (ultra-ristretto per M1)
SEARCH_SPACE = {
    "lr0": [0.0008, 0.001],
    "cls": [1.0, 1.2, 1.5],
    "box": [5.0, 6.0],
    "mosaic": [0.8, 1.0],
}

# File per salvare i risultati incrementali
RESULTS_FILE = SAVE_DIR / "results_incremental.csv"
BEST_PARAMS_FILE = SAVE_DIR / "best_params_so_far.json"
LOG_FILE = SAVE_DIR / "grid_search_log.txt"

def log_message(msg):
    """Scrive un messaggio nel log e lo stampa"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {msg}"
    print(log_entry)
    with open(LOG_FILE, 'a') as f:
        f.write(log_entry + "\n")

def save_intermediate_results(results_list, best_params):
    """Salva i risultati intermedi su disco"""
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(RESULTS_FILE, index=False)
        log_message(f"💾 Risultati salvati: {len(results_list)} trial completati")
    
    with open(BEST_PARAMS_FILE, 'w') as f:
        json.dump(best_params, f, indent=2)
        log_message(f"⭐ Best params so far: {best_params}")

def print_current_best(results_list):
    """Stampa i migliori risultati finora"""
    if not results_list:
        return
    
    df = pd.DataFrame(results_list)
    best_idx = df['fitness'].idxmax() if 'fitness' in df.columns else 0
    
    log_message("\n" + "="*60)
    log_message("📊 MIGLIOR RISULTATO FINORA:")
    if 'fitness' in df.columns:
        best_row = df.iloc[best_idx]
        log_message(f"   Fitness: {best_row['fitness']:.4f}")
        if 'lr0' in best_row:
            log_message(f"   lr0: {best_row['lr0']}, cls: {best_row['cls']}, box: {best_row['box']}, mosaic: {best_row['mosaic']}")
    log_message("="*60 + "\n")

def run_grid_search():
    """Esegue la grid search con salvataggio incrementale"""
    
    log_message("="*80)
    log_message("🚀 GRID SEARCH PER IPERPARAMETRI YOLO - CON MONITORAGGIO")
    log_message("="*80)
    log_message(f"Dataset: {DATA_PATH}")
    log_message(f"Spazio di ricerca: {SEARCH_SPACE}")
    total_combinations = len(SEARCH_SPACE['lr0']) * len(SEARCH_SPACE['cls']) * len(SEARCH_SPACE['box']) * len(SEARCH_SPACE['mosaic'])
    log_message(f"Totale combinazioni: {total_combinations}")
    log_message(f"Salvataggio incrementale in: {SAVE_DIR}")
    log_message("="*80)
    
    # Inizializza modello
    model = YOLO("yolo11n.pt")
    
    # Lista per accumulare i risultati
    all_results = []
    best_fitness = -float('inf')
    best_params = {}
    
    # Carica risultati precedenti se esistono (per riprendere da dove si era interrotto)
    if RESULTS_FILE.exists():
        existing_df = pd.read_csv(RESULTS_FILE)
        all_results = existing_df.to_dict('records')
        log_message(f"📂 Caricati {len(all_results)} trial precedenti")
        
        # Trova il best fitness precedente
        if 'fitness' in existing_df.columns:
            best_idx = existing_df['fitness'].idxmax()
            best_fitness = existing_df.iloc[best_idx]['fitness']
            for param in ['lr0', 'cls', 'box', 'mosaic']:
                if param in existing_df.columns:
                    best_params[param] = existing_df.iloc[best_idx][param]
            log_message(f"⭐ Best fitness precedente: {best_fitness:.4f}")
    
    # Genera tutte le combinazioni (se non vuoi usare il tuner automatico)
    # Oppure usa il tuner integrato ma con callback
    from itertools import product
    
    combinations = list(product(
        SEARCH_SPACE['lr0'],
        SEARCH_SPACE['cls'],
        SEARCH_SPACE['box'],
        SEARCH_SPACE['mosaic']
    ))
    
    # Filtra le combinazioni già fatte
    completed_keys = set()
    for r in all_results:
        key = (r.get('lr0'), r.get('cls'), r.get('box'), r.get('mosaic'))
        completed_keys.add(key)
    
    remaining = [c for c in combinations if c not in completed_keys]
    log_message(f"🔍 Combinazioni rimanenti: {len(remaining)} / {total_combinations}")
    
    # Esegui ogni trial manualmente per avere controllo incrementale
    for idx, (lr0_val, cls_val, box_val, mosaic_val) in enumerate(remaining):
        log_message(f"\n🔄 Trial {len(all_results)+1}/{total_combinations}")
        log_message(f"   Parametri: lr0={lr0_val}, cls={cls_val}, box={box_val}, mosaic={mosaic_val}")
        
        # Prepara i parametri per questo trial
        train_params = {
            "data": DATA_PATH,
            "model": "yolo11n.pt",
            "epochs": BASE_PARAMS["epochs"],
            "batch": BASE_PARAMS["batch"],
            "workers": BASE_PARAMS["workers"],
            "imgsz": BASE_PARAMS["imgsz"],
            "lr0": lr0_val,
            "cls": cls_val,
            "box": box_val,
            "mosaic": mosaic_val,
            "fliplr": BASE_PARAMS["fliplr"],
            "optimizer": BASE_PARAMS["optimizer"],
            "patience": BASE_PARAMS["patience"],
            "name": f"trial_{len(all_results)+1}",
            "exist_ok": True,
            "verbose": False,  # Riduci output per velocità
        }
        
        try:
            # Esegui training per questo trial
            results = model.train(**train_params)
            
            # Estrai metrica finale (mAP50-95 è la fitness tipica)
            if hasattr(results, 'results_dict'):
                fitness = results.results_dict.get('metrics/mAP50-95(B)', 0)
            else:
                # Fallback: prova a leggere dal CSV dei risultati
                import csv
                csv_path = Path(f"/Users/antonio/runs/detect/trial_{len(all_results)+1}/results.csv")
                if csv_path.exists():
                    df_trial = pd.read_csv(csv_path)
                    fitness = df_trial['metrics/mAP50-95(B)'].iloc[-1] if 'metrics/mAP50-95(B)' in df_trial.columns else 0
                else:
                    fitness = 0
            
            # Salva risultato
            trial_result = {
                "trial": len(all_results) + 1,
                "lr0": lr0_val,
                "cls": cls_val,
                "box": box_val,
                "mosaic": mosaic_val,
                "fitness": fitness,
                "timestamp": datetime.now().isoformat()
            }
            all_results.append(trial_result)
            
            # Aggiorna best se necessario
            if fitness > best_fitness:
                best_fitness = fitness
                best_params = {
                    "lr0": lr0_val,
                    "cls": cls_val,
                    "box": box_val,
                    "mosaic": mosaic_val,
                    "fitness": fitness
                }
                log_message(f"   🎉 NUOVO BEST! Fitness={fitness:.4f}")
            
            # Salva incrementale dopo ogni trial
            save_intermediate_results(all_results, best_params)
            print_current_best(all_results)
            
        except Exception as e:
            log_message(f"   ❌ ERRORE: {e}")
            # Salva comunque l'errore
            error_result = {
                "trial": len(all_results) + 1,
                "lr0": lr0_val,
                "cls": cls_val,
                "box": box_val,
                "mosaic": mosaic_val,
                "fitness": -1,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            all_results.append(error_result)
            save_intermediate_results(all_results, best_params)
    
    # Risultati finali
    log_message("\n" + "="*80)
    log_message("✅ GRID SEARCH COMPLETATA!")
    log_message("="*80)
    log_message(f"📊 Migliori parametri trovati:")
    log_message(f"   lr0: {best_params.get('lr0')}")
    log_message(f"   cls: {best_params.get('cls')}")
    log_message(f"   box: {best_params.get('box')}")
    log_message(f"   mosaic: {best_params.get('mosaic')}")
    log_message(f"   Fitness (mAP50-95): {best_fitness:.4f}")
    log_message(f"\n📁 Risultati salvati in: {SAVE_DIR}")
    log_message(f"   - {RESULTS_FILE}")
    log_message(f"   - {BEST_PARAMS_FILE}")
    log_message(f"   - {LOG_FILE}")
    
    return all_results, best_params

if __name__ == "__main__":
    results, best = run_grid_search()
