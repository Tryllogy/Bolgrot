# Déploiement — Oracle Cloud (VM Always Free, gratuit à vie)

Le jeu + l'IA + le front tournent dans **un seul** conteneur Docker (FastAPI sert
`web/dist` et l'API sur la même origine — pas de CORS). On l'héberge sur une **VM
Oracle Ampere (ARM) Always Free** (**2 OCPU / 12 Go RAM**, allumée 24/7, aucun
cold start), avec **Caddy** en reverse proxy pour l'**HTTPS automatique** et un
**sous-domaine gratuit DuckDNS** (Let's Encrypt ne certifie pas une IP nue).

```
Internet ──HTTPS──> Caddy (:443, cert Let's Encrypt auto)
                      └─reverse_proxy─> conteneur Docker (127.0.0.1:7860, uvicorn)
```

## Fichiers de déploiement (déjà dans le repo)

- `Dockerfile` — multi-stage : (1) `node` build du front → `web/dist`,
  (2) runtime `python:3.12-slim`. **torch arch-aware** : sur ARM (notre VM) il vient
  de PyPI (déjà CPU-only) ; sur x86_64 il vient de l'index CPU (pas de CUDA). User
  UID 1000, `$PORT` honoré (défaut 7860).
- `requirements-deploy.txt`, `.dockerignore` — deps runtime + contexte minimal.
- `src/ai/az.pt` — le modèle, **versionné** (39 Ko).

---

## 1. Créer la VM Always Free

Compte : https://www.oracle.com/cloud/free/ (carte requise au signup, **aucun débit**
sur les ressources *Always Free*).

Console → **Compute → Instances → Create instance** :
- **Image** : Ubuntu 24.04 (ou 22.04).
- **Shape** : *Change shape* → **Ampere** → `VM.Standard.A1.Flex`. Mets
  **2 OCPU / 12 Go** = la **totalité** de l'allocation Always Free (Oracle l'a
  réduite en juin 2026 : c'était 4 OCPU/24 Go, désormais 2/12 pour le tier
  gratuit ; le 4/24 demande un compte Pay-As-You-Go, avec risque de facturation
  hors quotas). 2/12 suffit très largement pour Bolgrot.
- **SSH keys** : ajoute ta clé publique (`~/.ssh/id_ed25519.pub`).
- Note l'**IP publique** attribuée.

> ⚠️ Si « Out of capacity » sur Ampere (fréquent, surtout si ta région n'a qu'1
> *Availability Domain*) : ce n'est pas une erreur de ta part, le stock de serveurs
> ARM gratuits est épuisé à cet instant. Réessaie en **heures creuses** (2-6 h du
> matin, heure de la région), ou lance le **script de retry automatique**
> `./oci-retry.sh` (il tente toutes les 60 s et crée la VM dès qu'une place se
> libère). Prérequis du script : **OCI CLI** installé + configuré :
> ```bash
> bash -c "$(curl -L https://raw.githubusercontent.com/oracle/oci-cli/master/scripts/install/install.sh)"
> oci setup config   # génère une paire de clés API ; colle la clé PUBLIQUE affichée
>                     # dans Console → (avatar) → My profile → API keys → Add API key
> oci iam region list  # test : doit renvoyer du JSON => CLI OK
> ```
> Renseigne `SSH_PUB_KEY` en tête de `oci-retry.sh` (défaut `~/.ssh/pub_key.pub`),
> puis `./oci-retry.sh`. Il auto-détecte AD / subnet public / image Ubuntu ARM.

## 2. Ouvrir les ports 80 et 443 (DEUX niveaux — piège Oracle classique)

**a) Pare-feu cloud (Security List)** : VCN → Subnet → Security List → *Add Ingress
Rules* : source `0.0.0.0/0`, TCP, ports **80** puis **443**.

**b) Pare-feu de l'instance** : les images Ubuntu d'Oracle bloquent tout sauf SSH
via `iptables`. En SSH sur la VM :
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save     # persiste au reboot
```

## 3. Sous-domaine gratuit (DuckDNS)

https://www.duckdns.org (login GitHub/Google) → crée p. ex. `bolgrot.duckdns.org`
et mets l'**IP publique de la VM** dans le champ *current ip* → *update*.
(Un domaine à toi marche aussi : fais pointer un enregistrement A vers l'IP.)

## 4. Installer Docker sur la VM

```bash
ssh ubuntu@<IP_PUBLIQUE>
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu       # docker sans sudo (reconnecte-toi ensuite)
```

## 5. Récupérer le code + builder l'image

Le plus simple : pousser le repo sur GitHub puis `git clone` sur la VM (sinon
`scp -r` le projet). Puis, à la racine du projet **sur la VM** :
```bash
git clone https://github.com/<toi>/Bolgrot.git && cd Bolgrot
docker build -t bolgrot .            # build natif ARM sur la VM (~5-10 min)
```
> Le build inclut un `python -c "import torch, ..., pygame"` qui échoue tôt et
> clairement si une lib manque.

## 6. Lancer le conteneur

```bash
docker run -d --name bolgrot --restart unless-stopped \
  -p 127.0.0.1:7860:7860 \
  -e AZ_POOL_WORKERS=1 \
  bolgrot
```
- `--restart unless-stopped` : redémarre au reboot de la VM.
- `-p 127.0.0.1:7860:...` : exposé **uniquement en local** ; c'est Caddy (étape 7)
  qui expose au monde en HTTPS.
- `AZ_POOL_WORKERS` : nb de recherches AZ en parallèle = **nb d'OCPU**. Mets **1**
  sur une VM 1 OCPU (capacité gratuite fréquente), **2** si tu as 2 OCPU. Chaque
  hint est 1 worker CPU-bound ; inutile de dépasser le nb de cœurs.

Test local sur la VM : `curl -s localhost:7860/map | head -c 60` doit répondre.

## 7. HTTPS automatique avec Caddy

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
  | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
  | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg
sudo chmod o+r /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```
Édite `/etc/caddy/Caddyfile` pour qu'il contienne **uniquement** :
```
bolgrot.duckdns.org {
    reverse_proxy 127.0.0.1:7860
}
```
Puis :
```bash
sudo systemctl reload caddy
```
Caddy obtient tout seul un certificat Let's Encrypt (ports 80/443 ouverts requis).

## 8. Vérifier

Ouvre **https://bolgrot.duckdns.org** → le jeu doit s'afficher (cadenas HTTPS), et
le bouton *Indice* doit renvoyer un coup.

---

## Mettre à jour l'app

```bash
cd Bolgrot && git pull
docker build -t bolgrot .
docker rm -f bolgrot
docker run -d --name bolgrot --restart unless-stopped \
  -p 127.0.0.1:7860:7860 -e AZ_POOL_WORKERS=1 bolgrot
```

## Notes

- **Ressources** : torch ≈ 700 Mo disque + quelques centaines de Mo en RAM ; le
  réseau AZ est un petit MLP (~40 Ko/worker). Large sous 12 Go. Le CPU est le
  facteur limitant d'un hint (ARM Ampere s'en sort bien ; hint 1000 sims ~1-3 s).
- **Budget de recherche** : le sélecteur du front (300/1000/8000) reste dispo ;
  l'escalade 8000 (« réessayer plus fort ») est plus lente, normal.
- **Sécurité** : garde le conteneur bindé sur `127.0.0.1` (jamais `0.0.0.0`), seul
  Caddy est exposé. Pense aux mises à jour système (`sudo apt upgrade`).
- **DuckDNS + IP** : l'IP Always Free est stable tant que l'instance vit ; si tu la
  recrées, remets à jour l'IP sur DuckDNS.

# Déploiement — Google Cloud Run (alternative à Oracle, sans VM)

Si Oracle reste bloqué en « Out of capacity », Cloud Run héberge la **même image
Docker** sans gérer de serveur : Cloud Build build depuis le repo, Cloud Run sert
en HTTPS (domaine `*.run.app` fourni, certificat auto — pas de Caddy ni DuckDNS).
**Scale-to-zero** : gratuit au repos, mais **cold start ~5-10 s au 1er hint** après
inactivité (torch se recharge), puis chaud. CPU **plein pendant la requête** → le
hint lui-même reste rapide (1-3 s à 1000 sims).

```
Internet ──HTTPS (*.run.app, cert auto)──> Cloud Run ──$PORT──> conteneur (uvicorn)
```

## 1. Compte + CLI

Compte : https://console.cloud.google.com (carte requise, **aucun débit** dans les
quotas Always Free). Installe le CLI `gcloud` (SDK), puis, **toi** (étapes
interactives navigateur — lance-les avec `! ` depuis le prompt) :
```bash
gcloud auth login                       # ouvre le navigateur
gcloud projects create bolgrot-app --name=Bolgrot   # ou réutilise un projet
gcloud config set project bolgrot-app
gcloud auth application-default login    # creds pour Cloud Build
```
Active la **facturation** sur le projet (Console → Billing → lier un compte de
facturation) — obligatoire même pour l'Always Free, sinon les APIs refusent.

## 2. Activer les APIs

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com
```

## 3. Déployer (build + run en une commande)

À la racine du projet (le `.dockerignore` limite le contexte envoyé) :
```bash
gcloud run deploy bolgrot \
  --source . \
  --region europe-west9 \
  --allow-unauthenticated \
  --cpu 2 --memory 2Gi \
  --timeout 300 \
  --set-env-vars AZ_POOL_WORKERS=2
```
- `--source .` : Cloud Build build le `Dockerfile` (x86_64 → torch CPU, ~700 Mo)
  et pousse l'image dans Artifact Registry, sans Docker local.
- `--region europe-west9` : Paris (latence mini depuis la France). `europe-west1`
  (Belgique) marche aussi.
- `--cpu 2 --memory 2Gi` : torch + le pool AZ ont besoin de marge ; 512 Mo (défaut)
  fait OOM. `AZ_POOL_WORKERS=2` = les 2 vCPU alloués (hints parallèles / escalade).
- `--allow-unauthenticated` : accès public (c'est un jeu). CPU **facturé seulement
  pendant les requêtes** (défaut Cloud Run) → gratuit au repos.

La commande affiche l'URL `https://bolgrot-xxxx.run.app` à la fin.

## 4. Vérifier

Ouvre l'URL `*.run.app` → le jeu s'affiche (HTTPS), le bouton *Indice* renvoie un
coup (le **tout premier** peut prendre ~10 s = cold start, ensuite instantané).

## Mettre à jour · options

- **Redéployer** : `git push` puis relancer la commande `gcloud run deploy` du §3
  (ou brancher un trigger Cloud Build sur le repo GitHub pour l'auto-deploy).
- **Zéro cold start** (payant, petit) : `--min-instances 1` garde un conteneur
  chaud en permanence — la seule chose non gratuite ; à réserver si l'UX du 1er
  hint gêne.
- **Sur-souscription CPU** : à fort trafic, ajoute `--concurrency 4` pour éviter
  que trop de hints CPU-bound se partagent les 2 vCPU (inutile en usage perso).

---

## Autres hébergeurs (le Dockerfile est portable)

L'image marche aussi ailleurs sans modif (port `$PORT` géré, torch arch-aware) :
- **Fly.io** — Docker natif, machine épinglée **toujours chaude** (aucun cold
  start) ; ~5$/mois (plus de vrai tier gratuit depuis fin 2024). Le choix « UX
  maximale » si les quelques euros ne gênent pas.
- **Render** — déploie direct depuis GitHub (zéro CLI) mais **spin-down** après
  15 min (cold start ~30 s) et 512 Mo RAM (risque OOM torch). Le plus simple, l'UX
  hint la moins bonne.
- **Hugging Face** — nécessite un plan **PRO payant** (~9$/mois) pour les Spaces
  Docker depuis 2026 (le tier gratuit ne fait plus tourner de backend Docker).
