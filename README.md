# MMS-TTS Baoulé — Synthèse vocale pour le baoulé de Côte d'Ivoire

Adaptation du modèle multilingue **MMS-TTS** de Meta pour générer de la voix synthétique en **baoulé** (langue kwa parlée en Côte d'Ivoire, code ISO 639-3 `bci`), une langue non couverte nativement par MMS.

> 🎧 Modèle final : [`Tree-AI-lab/mms-tts-bau-finetuned`](https://huggingface.co/Tree-AI-lab/mms-tts-bau-finetuned)

---

## Sommaire

- [Contexte et objectif](#contexte-et-objectif)
- [Vue d'ensemble du pipeline](#vue-densemble-du-pipeline)
- [Ressources produites](#ressources-produites)
- [Structure du dépôt](#structure-du-dépôt)
- [1. Pourquoi partir de l'akan (`facebook/mms-tts-aka`)](#1-pourquoi-partir-de-lakan-facebookmms-tts-aka)
- [2. Le dataset : `google/WaxalNLP` (sous-config `bau_tts`)](#2-le-dataset--googlewaxalnlp-sous-config-bau_tts)
- [3. Le vocabulaire baoulé n'est pas le vocabulaire akan](#3-le-vocabulaire-baoulé-nest-pas-le-vocabulaire-akan)
- [4. Correction du checkpoint donneur](#4-correction-du-checkpoint-donneur)
- [5. Fine-tuning avec `finetune-hf-vits`](#5-fine-tuning-avec-finetune-hf-vits)
- [6. Robustesse de l'entraînement sur Colab](#6-robustesse-de-lentraînement-sur-colab)
- [Reproduire ce projet](#reproduire-ce-projet)
- [Limites connues et pistes d'amélioration](#limites-connues-et-pistes-damélioration)
- [Licence](#licence)
- [Remerciements et sources](#remerciements-et-sources)

---

## Contexte et objectif

Le baoulé ne fait pas partie des ~1100 langues couvertes nativement par [MMS-TTS](https://huggingface.co/facebook/mms-tts#supported-languages) (Massively Multilingual Speech, Meta). Ce projet construit un modèle de synthèse vocale baoulé en **adaptant** un checkpoint MMS d'une langue proche (l'akan du Ghana) via un fine-tuning sur des données réelles de baoulé ivoirien, plutôt qu'en entraînant un modèle from scratch.

Le projet s'appuie sur le dépôt public [`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits), qui fournit le script d'entraînement (`run_vits_finetuning.py`) et l'outillage de conversion de checkpoint pour fine-tuner des modèles VITS/MMS avec les outils Hugging Face.

## Vue d'ensemble du pipeline

```text
facebook/mms-tts-aka (générateur akan préentraîné, Meta)
  + discriminateur Meta (full_models/aka/D_100000.pth)
  │
  ├─► conversion (pad_token_id patché) ──► Tree-AI-lab/mms-tts-bau-baseline
  │
google/WaxalNLP (bau_tts)                  Tree-AI-lab/mms-tts-bau-baseline
  │                                                    │
  ├─► nettoyage texte                                  │
  ├─► filtrage locuteur (JH)                            │
  ├─► construction vocabulaire baoulé ─┐                │
  │                                    ▼                ▼
  │                         Tree-AI-lab/mms-tts-bau-tokenizer
  │                                    │
  ▼                                    ▼
Tree-AI-lab/bau-tts-monospeaker ──► transfert d'embeddings caractère par caractère
                                                    │
                                                    ▼
                                    fine-tuning (finetune-hf-vits)
                                                    │
                                                    ▼
                                Tree-AI-lab/mms-tts-bau-finetuned
```

Le pipeline est organisé en **deux notebooks Google Colab** aux responsabilités séparées :

| | `pipeline-tts_baoulé.ipynb` (préparation) | `training_mms_tts_baoule.ipynb` (entraînement) |
|---|---|---|
| **Rôle** | Explore `google/WaxalNLP` (`bau_tts`), nettoie le texte, filtre le locuteur JH, construit le vocabulaire baoulé, convertit le checkpoint donneur akan | Charge ces ressources déjà publiées, transfère les embeddings, lance et reprend l'entraînement, publie le modèle final |
| **Produit** | `Tree-AI-lab/bau-tts-monospeaker`, `Tree-AI-lab/mms-tts-bau-tokenizer`, `Tree-AI-lab/mms-tts-bau-baseline` | `Tree-AI-lab/mms-tts-bau-finetuned` |
| **Consomme** | `facebook/mms-tts-aka`, `google/WaxalNLP` | Les 3 dépôts publiés par le notebook de préparation |

## Ressources produites

| Dépôt Hugging Face | Contenu |
|---|---|
| [`Tree-AI-lab/bau-tts-monospeaker`](https://huggingface.co/datasets/Tree-AI-lab/bau-tts-monospeaker) | Sous-ensemble de `bau_tts` : texte nettoyé, filtré sur le locuteur le mieux représenté (JH, 276 échantillons train / 30 validation / 28 test) |
| [`Tree-AI-lab/mms-tts-bau-tokenizer`](https://huggingface.co/Tree-AI-lab/mms-tts-bau-tokenizer) | Tokenizer VITS dédié au baoulé, vocabulaire construit à partir du corpus réel (voir §3) |
| [`Tree-AI-lab/mms-tts-bau-baseline`](https://huggingface.co/Tree-AI-lab/mms-tts-bau-baseline) | Checkpoint donneur akan + discriminateur, prêt pour le fine-tuning |
| [`Tree-AI-lab/mms-tts-bau-finetuned`](https://huggingface.co/Tree-AI-lab/mms-tts-bau-finetuned) | **Modèle final** — synthèse vocale baoulé |

## Structure du dépôt

```
.
├── README.md
├── pipeline-tts_baoulé.ipynb          # Préparation : dataset, tokenizer, checkpoint donneur
└── training_mms_tts_baoule.ipynb      # Entraînement : fine-tuning robuste sur Colab GPU
```

---

## 1. Pourquoi partir de l'akan (`facebook/mms-tts-aka`)

Meta n'a pas publié de checkpoint `facebook/mms-tts-bci` (le baoulé n'est pas dans la couverture native de MMS-TTS). Le baoulé appartient au sous-groupe **Anyi-Baoulé** de la famille Kwa, proche de l'**akan** (Ghana) — le donneur linguistique le plus pertinent disponible dans MMS parmi les checkpoints existants (`fon`, `ewe` étant d'autres candidats plus éloignés).

Fine-tuner un checkpoint MMS existant plutôt que d'entraîner from scratch est la stratégie recommandée par `finetune-hf-vits` : avec le bon jeu de données, on peut obtenir un résultat correct en environ 20 minutes avec seulement 80 à 150 échantillons — la donneur n'a pas besoin d'être parfait, il sert de point de départ phonétique et architectural.

## 2. Le dataset : `google/WaxalNLP` (sous-config `bau_tts`)

Le sous-ensemble TTS baoulé de [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP) contient 1216 échantillons (972 train / 122 validation / 122 test), avec les colonnes `id`, `speaker_id`, `text`, `locale`, `gender`, `audio`.

**Choix du locuteur.** Le dataset contient plusieurs locuteurs (`JH`, `RK`, `KK`, `JK`...). Pour un premier modèle propre, on isole le locuteur le mieux représenté (**JH**, 276 échantillons train), au lieu d'entraîner sur un mélange de voix — cohérent avec l'architecture MMS de base, conçue pour un seul locuteur (`num_speakers: 1`, `speaker_embedding_size: 0`).

**Filtrage par durée.** Les durées audio vont de 2,7 s à plus de 412 s (des paragraphes entiers lus d'un bloc). VITS calcule une matrice d'alignement texte/audio proportionnelle à la durée — un clip de plusieurs minutes fait exploser la mémoire GPU. On filtre à **≤ 30 secondes**, ce qui conserve 230 des 276 clips du locuteur JH.

**Nettoyage du texte.** Le texte brut contient du bruit typographique (espaces invisibles `\u200b`, guillemets/tirets typographiques multiples, emoji) mélangé à du contenu légitime (noms propres français intégrés au texte baoulé — *Félix Houphouet-Boigny*, *Hôtel Ivoire* — et nombres entre crochets réellement prononcés, ex. `[25]`, `[30]`). La fonction `clean_baoule_text` :
- normalise le texte en NFC,
- fusionne les variantes d'apostrophe et de guillemets vers une forme unique,
- retire les emoji et symboles décoratifs (`°`, `²`, `•`, `→`),
- **conserve** les phonèmes, noms propres, nombres entre crochets et marquage tonal.

## 3. Le vocabulaire baoulé n'est pas le vocabulaire akan

C'est le point technique le plus important du projet. Le tokenizer `facebook/mms-tts-aka` ne couvre que **30 caractères**. Or le code source du `VitsTokenizer` de `transformers` **supprime silencieusement** tout caractère absent du vocabulaire (pas d'erreur, pas de token `<unk>` — le caractère disparaît du texte avant l'entraînement) :

```python
filtered_text = "".join(list(filter(lambda char: char in self.encoder, filtered_text))).strip()
```

Le baoulé utilise des caractères absents du vocabulaire akan — notamment `ɲ`, `ɡ`, `ʁ`, `ʃ`, le marquage tonal (accent combinant sur `ɛ`/`ɔ`), sans compter la ponctuation et les chiffres. Sans correction, l'entraînement produirait un désalignement texte/audio silencieux (le son existe, le texte correspondant a disparu).

**Solution : un tokenizer baoulé dédié.** Le vocabulaire est reconstruit à partir de l'intégralité du corpus `bau_tts` nettoyé (tous locuteurs confondus, pour être complet dès maintenant), avec `<pad>` (id 0, le token « blanc » que VITS insère entre chaque caractère) et `<unk>` comme tokens spéciaux dédiés. Publié sous `Tree-AI-lab/mms-tts-bau-tokenizer`.

**Transfert d'embeddings, pas réinitialisation totale.** Plutôt que de réinitialiser aléatoirement toute la couche d'embeddings de caractères (option `override_vocabulary_embeddings` du script officiel), le notebook d'entraînement copie individuellement l'embedding déjà appris par le modèle akan pour chaque caractère **commun** aux deux vocabulaires (`a`, `e`, `i`, `o`, `u`, `ɛ`, `ɔ`...), et ne réinitialise aléatoirement que les caractères réellement nouveaux au baoulé. Le modèle démarre donc avec un début d'apprentissage préservé sur les sons partagés entre les deux langues, au lieu de tout réapprendre depuis zéro.

## 4. Correction du checkpoint donneur

Le `config.json` de `facebook/mms-tts-aka` ne définit pas `pad_token_id` — une version récente de `transformers` lève une `AttributeError` au lieu de traiter ce champ comme optionnel, ce qui fait planter la construction du modèle (`VitsModelForPreTraining`). Le pipeline patche localement une copie du `config.json` (`pad_token_id: 0`) avant la conversion.

La conversion fusionne ensuite ce générateur patché avec le discriminateur original de Meta (`facebook/mms-tts`, `full_models/aka/D_100000.pth` — disponible uniquement dans les poids d'entraînement bruts, absent des checkpoints d'inférence HF classiques), via le script `convert_original_discriminator_checkpoint.py` de `finetune-hf-vits`. Résultat publié : `Tree-AI-lab/mms-tts-bau-baseline`.

## 5. Fine-tuning avec `finetune-hf-vits`

Le script `run_vits_finetuning.py` du dépôt [`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits) pilote l'entraînement GAN (générateur + discriminateur) via un fichier de configuration JSON. Paramètres clés utilisés :

- `model_name_or_path` / `tokenizer_name` : le checkpoint local reconstruit avec les embeddings transférés (§3)
- `dataset_name` : `Tree-AI-lab/bau-tts-monospeaker`
- `max_duration_in_seconds: 30`, `min_duration_in_seconds: 1.0`
- pondération des pertes GAN (`weight_disc`, `weight_gen`, `weight_kl`, `weight_mel`, `weight_duration`, `weight_fmaps`) : valeurs par défaut du dépôt, éprouvées sur d'autres langues MMS
- `fp16` désactivé côté entraînement final (FP32, plus stable pour ce premier run) au profit d'un `per_device_train_batch_size` réduit

## 6. Robustesse de l'entraînement sur Colab

Le notebook d'entraînement ajoute plusieurs couches de robustesse absentes du script d'origine, nécessaires pour un entraînement long sur un Colab gratuit (déconnexions fréquentes) :

- **Versions figées** (`transformers==4.44.2`, `pyarrow==17.0.0`, `datasets[audio]==3.6.0`...) pour éviter les régressions rencontrées avec des versions plus récentes (`pad_token_id` manquant, erreurs `ArrowInvalid: offset overflow` lors de l'écriture de gros datasets audio).
- **Checkpointing sur Google Drive** avec marqueur `COMPLETE` écrit uniquement après synchronisation confirmée du disque (`os.sync()` + délai), pour ne jamais reprendre depuis un checkpoint à moitié écrit.
- **Test de fumée** (2 pas sur 8 échantillons, dossier séparé) avant de lancer l'entraînement complet, pour détecter une erreur de configuration en quelques secondes plutôt qu'après plusieurs heures.
- **Détection et nettoyage automatique des checkpoints interrompus** avant reprise.
- **Révision de dataset figée** (`revision=data_info.sha`) pour garantir la reproductibilité même si le dataset évolue sur le Hub entre deux sessions.

---

## Reproduire ce projet

1. Ouvrir `pipeline-tts_baoulé.ipynb` dans Google Colab, l'exécuter dans l'ordre (installation → exploration du dataset → nettoyage/filtrage → construction du tokenizer → conversion du checkpoint donneur). Remplacer les identifiants de dépôt Hugging Face par les tiens.
2. Ouvrir `training_mms_tts_baoule.ipynb`, activer un GPU (Runtime Colab Python 3.12), monter Google Drive, adapter les constantes `DATASET` / `TOKENIZER` / `BASELINE` / `TARGET` en tête de notebook.
3. Exécuter jusqu'au test de fumée, vérifier qu'il passe, puis lancer l'entraînement complet.
4. Écouter un échantillon de validation et publier le modèle final.

### Prérequis

- Compte Hugging Face avec token en écriture
- GPU (Colab T4 suffit — le modèle ne fait que ~83M paramètres)
- Google Drive (pour la persistance des checkpoints entre sessions)

## Limites connues et pistes d'amélioration

- **V1 mono-locuteur uniquement** (voix JH) — une version multi-locuteur sur l'intégralité du dataset (`speaker_id_column_name` + `override_speaker_embeddings`) est en préparation pour ancrer davantage l'accent baoulé ivoirien et réduire la coloration akan actuelle.
- **Clips > 30 s exclus** (46 échantillons sur 276 pour le locuteur JH) — ce sont de vrais paragraphes longs, pas des artefacts ; une segmentation par alignement forcé (ex. Montreal Forced Aligner) permettrait de les réintégrer.
- Qualité non évaluée formellement (pas de MOS, pas de test d'écoute structuré à ce stade).

## Licence

Le modèle `facebook/mms-tts-aka` est distribué sous licence **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)** (non commerciale) — tout modèle dérivé, y compris `Tree-AI-lab/mms-tts-bau-finetuned`, hérite de cette licence. Le dataset `google/WaxalNLP` (`bau_tts`) est distribué sous licence CC-BY-4.0.

## Remerciements et sources

- [`ylacombe/finetune-hf-vits`](https://github.com/ylacombe/finetune-hf-vits) — script et outillage de fine-tuning VITS/MMS avec Hugging Face `transformers`
- [`facebook/mms-tts-aka`](https://huggingface.co/facebook/mms-tts-aka) — checkpoint donneur (Massively Multilingual Speech, Meta AI)
- [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP) — dataset source (sous-config `bau_tts`)
- [Documentation `transformers` — modèle VITS](https://huggingface.co/docs/transformers/model_doc/vits)