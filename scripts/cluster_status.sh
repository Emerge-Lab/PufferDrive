#!/bin/bash
# Quick script to check cluster node availability

echo "=== Node Availability by Partition ==="
echo ""
printf "%-20s %6s %6s %6s %6s %6s\n" "PARTITION" "IDLE" "MIXED" "ALLOC" "DOWN" "TOTAL"
echo "--------------------------------------------------------------"

sinfo -h -o "%P %T %D" | awk '
{
    part=$1
    state=$2
    count=$3

    # Remove * from default partition
    gsub(/\*/, "", part)

    if (state == "idle") idle[part] += count
    else if (state == "mixed") mixed[part] += count
    else if (state == "allocated") alloc[part] += count
    else if (state ~ /down|drain/) down[part] += count

    total[part] += count
}
END {
    for (p in total) {
        printf "%-20s %6d %6d %6d %6d %6d\n", p, idle[p]+0, mixed[p]+0, alloc[p]+0, down[p]+0, total[p]
    }
}' | sort

echo ""
echo "=== Summary ==="
sinfo -h -o "%T %D" | awk '
{
    state=$1
    count=$2
    if (state == "idle") idle += count
    else if (state == "mixed") mixed += count
    else if (state == "allocated") alloc += count
    total += count
}
END {
    printf "Total nodes: %d | Idle: %d | Mixed: %d | Allocated: %d\n", total, idle, mixed, alloc
}'

echo ""
echo "=== Your Jobs ==="
squeue -u $USER -o "%.8i %.12P %.20j %.2t %.10M %.4D %R" 2>/dev/null || echo "No jobs running"
