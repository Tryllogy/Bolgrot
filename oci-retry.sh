#!/usr/bin/env bash
# ============================================================================
#  oci-retry.sh — crée la VM Oracle Ampere (A1.Flex) Always Free EN BOUCLE
#  jusqu'à ce que la capacité ARM se libère, puis affiche l'IP publique.
#
#  Prérequis : OCI CLI installé + configuré (~/.oci/config). Voir DEPLOY.md §1.
#  Usage     : ./oci-retry.sh
#  Arrêt     : Ctrl-C (ou automatique dès que la VM est créée).
# ============================================================================
set -uo pipefail

# ---- Réglages --------------------------------------------------------------
DISPLAY_NAME="bolgrot-vm"
SHAPE="VM.Standard.A1.Flex"
OCPUS=2                       # 2 OCPU / 12 Go = TOTALITÉ du Always Free (2026)
MEM_GB=12
OS="Canonical Ubuntu"
OS_VERSION="24.04"
SSH_PUB_KEY="$HOME/.ssh/pub_key.pub"   # <-- ta clé publique
BOOT_VOL_GB=50                # 50 Go (Always Free = 200 Go de block total)
RETRY_SECONDS=90              # attente entre deux tentatives (capacité/réseau)
THROTTLE_SECONDS=300          # attente plus longue si Oracle throttle ("Too many requests")

# OCIDs : laisse VIDE pour auto-détection, ou colle-les à la main si besoin.
COMPARTMENT_ID=""             # défaut = tenancy (compartiment racine)
AD_NAME=""                    # défaut = 1er Availability Domain
SUBNET_ID=""                  # défaut = 1er subnet PUBLIC du compartiment
IMAGE_ID=""                   # défaut = dernière image Ubuntu ARM
# ----------------------------------------------------------------------------

command -v oci >/dev/null || { echo "❌ OCI CLI absente. Voir DEPLOY.md §1."; exit 1; }
[ -f "$SSH_PUB_KEY" ] || { echo "❌ Clé SSH publique introuvable : $SSH_PUB_KEY"; exit 1; }

echo "🔎 Auto-détection des paramètres manquants…"

# Compartiment = tenancy (root) si non fourni : lu dans ~/.oci/config
if [ -z "$COMPARTMENT_ID" ]; then
    COMPARTMENT_ID=$(grep -E '^tenancy' ~/.oci/config | head -1 | cut -d= -f2 | tr -d ' ')
fi
[ -n "$COMPARTMENT_ID" ] || { echo "❌ Compartiment introuvable — renseigne COMPARTMENT_ID."; exit 1; }

# Availability Domain (tu n'en as qu'un : AD-1)
if [ -z "$AD_NAME" ]; then
    AD_NAME=$(oci iam availability-domain list \
        --compartment-id "$COMPARTMENT_ID" \
        --query 'data[0].name' --raw-output)
fi

# Subnet public (le 1er trouvé). Doit être PUBLIC pour avoir une IP publique.
if [ -z "$SUBNET_ID" ]; then
    SUBNET_ID=$(oci network subnet list \
        --compartment-id "$COMPARTMENT_ID" \
        --query "data[?\"prohibit-public-ip-on-vnic\"==\`false\`] | [0].id" \
        --raw-output)
fi
[ -n "$SUBNET_ID" ] && [ "$SUBNET_ID" != "null" ] || {
    echo "❌ Aucun subnet PUBLIC trouvé. Crée une VCN 'with Internet Connectivity'"
    echo "   puis renseigne SUBNET_ID à la main."; exit 1; }

# Dernière image Ubuntu 24.04 compatible A1.Flex (ARM)
if [ -z "$IMAGE_ID" ]; then
    IMAGE_ID=$(oci compute image list \
        --compartment-id "$COMPARTMENT_ID" \
        --operating-system "$OS" \
        --operating-system-version "$OS_VERSION" \
        --shape "$SHAPE" \
        --sort-by TIMECREATED --sort-order DESC \
        --query 'data[0].id' --raw-output)
fi
[ -n "$IMAGE_ID" ] && [ "$IMAGE_ID" != "null" ] || {
    echo "❌ Image Ubuntu ARM introuvable — renseigne IMAGE_ID."; exit 1; }

cat <<EOF

📋 Paramètres retenus :
   AD          : $AD_NAME
   Compartiment: ${COMPARTMENT_ID:0:25}…
   Subnet      : ${SUBNET_ID:0:25}…
   Image       : ${IMAGE_ID:0:25}…
   Shape       : $SHAPE  ($OCPUS OCPU / $MEM_GB Go)
   Nom         : $DISPLAY_NAME
   Retry       : toutes les ${RETRY_SECONDS}s jusqu'au succès

EOF

attempt=0
while true; do
    attempt=$((attempt + 1))
    printf '⏳ Tentative %d (%s)… ' "$attempt" "$(date '+%H:%M:%S')"

    if OUT=$(oci compute instance launch \
            --availability-domain "$AD_NAME" \
            --compartment-id "$COMPARTMENT_ID" \
            --shape "$SHAPE" \
            --shape-config "{\"ocpus\":$OCPUS,\"memoryInGBs\":$MEM_GB}" \
            --image-id "$IMAGE_ID" \
            --subnet-id "$SUBNET_ID" \
            --assign-public-ip true \
            --boot-volume-size-in-gbs "$BOOT_VOL_GB" \
            --display-name "$DISPLAY_NAME" \
            --ssh-authorized-keys-file "$SSH_PUB_KEY" \
            --wait-for-state RUNNING 2>&1); then
        echo "✅ SUCCÈS !"
        INSTANCE_ID=$(echo "$OUT" | grep -oE '"id": "ocid1.instance[^"]+"' | head -1 | cut -d'"' -f4)
        echo "   Instance : $INSTANCE_ID"
        echo "🌐 Récupération de l'IP publique…"
        IP=$(oci compute instance list-vnics --instance-id "$INSTANCE_ID" \
                --query 'data[0]."public-ip"' --raw-output 2>/dev/null)
        echo ""
        echo "════════════════════════════════════════════════"
        echo "  IP PUBLIQUE : $IP"
        echo "  SSH         : ssh -i ~/.ssh/pub_key ubuntu@$IP"
        echo "════════════════════════════════════════════════"
        echo "  Suite : DEPLOY.md §2 (ports) → §8."
        exit 0
    fi

    # Erreurs VRAIMENT bloquantes (ne se résoudront pas toutes seules) : on stoppe.
    # Tout le reste (out of capacity, timeout réseau, 429/5xx…) : on réessaie.
    if echo "$OUT" | grep -qiE "NotAuthenticated|NotAuthorized|LimitExceeded|QuotaExceeded|CannotParseRequest|InvalidParameter|already exists"; then
        echo "⚠️  Erreur bloquante (config/quota) — arrêt :"
        echo "$OUT" | tail -20
        echo "→ À corriger avant de relancer (ce n'est pas un souci de capacité)."
        exit 1
    elif echo "$OUT" | grep -qiE "too many requests|toomanyrequests|429"; then
        # Oracle nous throttle : on attend PLUS longtemps pour se faire oublier.
        echo "throttling Oracle, pause de ${THROTTLE_SECONDS}s."
        sleep "$THROTTLE_SECONDS"
        continue
    elif echo "$OUT" | grep -qi "out of capacity\|out of host capacity"; then
        echo "capacité pleine, on réessaie dans ${RETRY_SECONDS}s."
    else
        # Transitoire (timeout réseau, 5xx…) : on réessaie aussi.
        REASON=$(echo "$OUT" | grep -iE 'message"|timed out|timeout' | head -1 | sed 's/^[[:space:]]*//')
        echo "erreur transitoire (${REASON:-réseau}), on réessaie dans ${RETRY_SECONDS}s."
    fi
    sleep "$RETRY_SECONDS"
done
