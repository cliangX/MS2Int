#!/bin/bash

ASCP="$HOME/miniconda3/envs/lc_ms/bin/ascp"
ASCP_KEY="$HOME/miniconda3/envs/lc_ms/etc/asperaweb_id_dsa.openssh"
ASCP_HOST="prd_ascp@fasp.ebi.ac.uk"

OUTDIR_BASE="/mnt/data_nas/jiaao/Immunopeptidome_data/tumor/T_Acute_Lymphoblastic_Leukemia"

# ── 通用下载函数（aspera，用于 PRIDE）────────────────────────────────────────
download_aspera() {
    local OUTDIR="$1"
    shift
    local URLS=("$@")

    mkdir -p "$OUTDIR"
    TOTAL=${#URLS[@]}
    COUNT=0; SKIP=0; FAIL=0

    for URL in "${URLS[@]}"; do
        FILENAME=$(basename "$URL")
        DEST="$OUTDIR/$FILENAME"
        if [ -f "$DEST" ] && [ -s "$DEST" ]; then
            COUNT=$((COUNT+1)); SKIP=$((SKIP+1))
            echo "[$(date '+%H:%M:%S')] SKIP ($COUNT/$TOTAL) $FILENAME"
            continue
        fi
        COUNT=$((COUNT+1))
        echo "[$(date '+%H:%M:%S')] DOWN ($COUNT/$TOTAL) $FILENAME"
        REMOTE_PATH=$(echo "$URL" | sed 's|http://ftp.pride.ebi.ac.uk||')
        $ASCP -TQ -l 500m -P 33001 -i "$ASCP_KEY" "$ASCP_HOST:$REMOTE_PATH" "$DEST"
        if [ $? -ne 0 ]; then
            FAIL=$((FAIL+1))
            echo "[$(date '+%H:%M:%S')] FAIL ($COUNT/$TOTAL) $FILENAME"
            rm -f "$DEST"
        fi
    done
    echo ""
    echo "====================================="
    echo "目录: $OUTDIR | 完成: $TOTAL | 跳过: $SKIP | 失败: $FAIL"
    echo "====================================="
}

# ════════════════════════════════════════════════════════════════════════════
# PXD011723 - T-ALL Jurkat 细胞系（4种裂解条件 × 2生物重复 × Even/Odd分组，共32个文件）
# 条件：C076（对照）、Ig01（IgG对照）、Triton01、NaD025
# 注：A375（黑色素瘤）不在此数据集中，所有文件均为 Jurkat T-ALL
# ════════════════════════════════════════════════════════════════════════════
B11723="http://ftp.pride.ebi.ac.uk/pride/data/archive/2019/01/PXD011723"
PXD011723_URLS=(
    # C076 - 生物重复1
    ${B11723}/FL605_IPP75_AN_Jurkat_C076_1BR_Even.raw
    ${B11723}/FL605_IPP75_AN_Jurkat_C076_1BR_Odd.raw
    ${B11723}/FL605b_IPP75_AN_Jurkat_C076_1BR_Even.raw
    ${B11723}/FL605b_IPP75_AN_Jurkat_C076_1BR_Odd.raw
    # Ig01 - 生物重复1
    ${B11723}/FL607_IPP75_AN_Jurkat_Ig01_1BR_Even.raw
    ${B11723}/FL607_IPP75_AN_Jurkat_Ig01_1BR_Odd.raw
    ${B11723}/FL607b_IPP75_AN_Jurkat_Ig01_1BR_Even.raw
    ${B11723}/FL607b_IPP75_AN_Jurkat_Ig01_1BR_Odd.raw
    # Triton01 - 生物重复1
    ${B11723}/FL609_IPP75_AN_Jurkat_Triton01_1BR_Even.raw
    ${B11723}/FL609_IPP75_AN_Jurkat_Triton01_1BR_Odd.raw
    ${B11723}/FL609b_IPP75_AN_Jurkat_Triton01_1BR_Even.raw
    ${B11723}/FL609b_IPP75_AN_Jurkat_Triton01_1BR_Odd.raw
    # NaD025 - 生物重复1
    ${B11723}/FL610_IPP75_AN_Jurkat_NaD025_1BR_Even.raw
    ${B11723}/FL610_IPP75_AN_Jurkat_NaD025_1BR_Odd.raw
    ${B11723}/FL610b_IPP75_AN_Jurkat_NaD025_1BR_Even.raw
    ${B11723}/FL610b_IPP75_AN_Jurkat_NaD025_1BR_Odd.raw
    # C076 - 生物重复2
    ${B11723}/FL613_IPP75_AN_Jurkat_C076_2BR_Even.raw
    ${B11723}/FL613_IPP75_AN_Jurkat_C076_2BR_Odd.raw
    ${B11723}/FL613b_IPP75_AN_Jurkat_C076_2BR_Even.raw
    ${B11723}/FL613b_IPP75_AN_Jurkat_C076_2BR_Odd.raw
    # Ig01 - 生物重复2
    ${B11723}/FL615_IPP75_AN_Jurkat_Ig01_2BR_Even.raw
    ${B11723}/FL615_IPP75_AN_Jurkat_Ig01_2BR_Odd.raw
    ${B11723}/FL615b_IPP75_AN_Jurkat_Ig01_2BR_Even.raw
    ${B11723}/FL615b_IPP75_AN_Jurkat_Ig01_2BR_Odd.raw
    # Triton01 - 生物重复2
    ${B11723}/FL617_IPP75_AN_Jurkat_Triton01_2BR_Even.raw
    ${B11723}/FL617_IPP75_AN_Jurkat_Triton01_2BR_Odd.raw
    ${B11723}/FL617b_IPP75_AN_Jurkat_Triton01_2BR_Even.raw
    ${B11723}/FL617b_IPP75_AN_Jurkat_Triton01_2BR_Odd.raw
    # NaD025 - 生物重复2
    ${B11723}/FL618_IPP75_AN_Jurkat_NaD025_2BR_Even.raw
    ${B11723}/FL618_IPP75_AN_Jurkat_NaD025_2BR_Odd.raw
    ${B11723}/FL618b_IPP75_AN_Jurkat_NaD025_2BR_Even.raw
    ${B11723}/FL618b_IPP75_AN_Jurkat_NaD025_2BR_Odd.raw
)

# ════════════════════════════════════════════════════════════════════════════
# PXD024562 - T-ALL LOUCY 细胞系（2个重复，共2个文件）
# 注：同数据集中的 A375 为黑色素瘤细胞系，已排除
# ════════════════════════════════════════════════════════════════════════════
B24562="http://ftp.pride.ebi.ac.uk/pride/data/archive/2021/06/PXD024562"
PXD024562_URLS=(
    ${B24562}/LOUCY_MHC_IP_5e8_R1.raw
    ${B24562}/LOUCY_MHC_IP_5e8_R2.raw
)

# ── 执行下载 ─────────────────────────────────────────────────────────────────
echo ">>> 开始下载 PXD011723 T-ALL（Jurkat 4条件×2重复，共32个文件）"
download_aspera "$OUTDIR_BASE/PXD011723" "${PXD011723_URLS[@]}"

echo ">>> 开始下载 PXD024562 T-ALL（LOUCY 细胞系，共2个文件）"
download_aspera "$OUTDIR_BASE/PXD024562" "${PXD024562_URLS[@]}"
