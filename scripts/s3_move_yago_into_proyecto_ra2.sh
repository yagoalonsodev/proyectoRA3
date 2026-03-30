#!/usr/bin/env bash
# Mueve todo lo que hay bajo yago_alonso/ (p. ej. raw/) dentro de yago_alonso/ProyectoRA2/
# Requiere: AWS CLI configurado (aws configure) o variables AWS_* en el entorno.
#
# Uso:
#   ./scripts/s3_move_yago_into_proyecto_ra2.sh
#
# Bucket y región según consola S3 (ej. París eu-west-3).

set -euo pipefail

BUCKET="${S3_BUCKET:-lasalle-bigdata-2025-2026}"
REGION="${AWS_REGION:-eu-west-3}"
SRC_PREFIX="yago_alonso"
DST_PREFIX="yago_alonso/ProyectoRA2"

echo "Bucket: $BUCKET  Region: $REGION"
echo "Origen: s3://${BUCKET}/${SRC_PREFIX}/"
echo "Destino base: s3://${BUCKET}/${DST_PREFIX}/"
echo ""
echo "Listando objetos bajo ${SRC_PREFIX}/ (excluyendo si ya existe ${DST_PREFIX})..."

# Lista objetos; si solo quieres mover 'raw', descomenta la línea siguiente y comenta el sync amplio:
# aws s3 mv "s3://${BUCKET}/${SRC_PREFIX}/raw/" "s3://${BUCKET}/${DST_PREFIX}/raw/" --recursive --region "$REGION"

# Opción segura: solo mover la carpeta raw/ si existe
if aws s3 ls "s3://${BUCKET}/${SRC_PREFIX}/raw/" --region "$REGION" 2>/dev/null | grep -q .; then
  echo "Moviendo ${SRC_PREFIX}/raw/ -> ${DST_PREFIX}/raw/ ..."
  aws s3 mv "s3://${BUCKET}/${SRC_PREFIX}/raw/" "s3://${BUCKET}/${DST_PREFIX}/raw/" --recursive --region "$REGION"
  echo "Hecho."
else
  echo "No hay objetos en s3://${BUCKET}/${SRC_PREFIX}/raw/ (o la ruta no existe). Nada que mover."
fi

echo ""
echo "Para el pipeline Polymarket, en tu .env usa:"
echo "  S3_BUCKET=${BUCKET}"
echo "  AWS_REGION=${REGION}"
echo "  S3_PREFIX=${DST_PREFIX}"
