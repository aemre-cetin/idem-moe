import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 8.5
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2

def create_fig1():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Main Accelerator Box 100
    ax.add_patch(patches.Rectangle((6, 10), 88, 80, fill=False, edgecolor='black', linewidth=1.5))
    ax.text(9, 87, "100: SPARSE MIXTURE-OF-EXPERTS (MoE) ACCELERATOR (GPU / NPU)", fontweight='bold', fontsize=10)

    # 102 Primary Memory (HBM)
    ax.add_patch(patches.Rectangle((10, 67), 80, 17, fill=False, edgecolor='black', linewidth=1.2))
    ax.text(12, 81, "102: PRIMARY ACCELERATOR MEMORY (HBM / VRAM)", fontweight='bold', fontsize=9.5)
    ax.text(12, 77.5, "104: Candidate Expert Token Embeddings Buffer [E, N, D]", fontsize=8.5, fontstyle='italic')
    
    e_labels = ["Expert 0 [N,D]", "Expert 1 [N,D]", "Expert 2 [N,D]", "Expert ...", "Expert E-1 [N,D]"]
    for i, lbl in enumerate(e_labels):
        x = 12 + i * 15.2
        ax.add_patch(patches.Rectangle((x, 69.5), 14.5, 6, fill=False, edgecolor='black', linewidth=0.9))
        ax.text(x + 7.25, 72.5, lbl, ha='center', va='center', fontsize=7.5)

    # Direct Memory Bus
    ax.annotate('', xy=(50, 67), xytext=(50, 58), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(52, 62, "Direct High-Speed Memory Bus (Zero cudaMalloc / Zero All-to-All Memory Spikes)", fontsize=8)

    # 110 Controller
    ax.add_patch(patches.Rectangle((10, 23), 80, 35, fill=False, edgecolor='black', linewidth=1.2))
    ax.text(12, 54.5, "110: HARDWARE MoE ROUTER & TOKEN CONSOLIDATION CONTROLLER", fontweight='bold', fontsize=9.5)

    # Sub-units
    ax.add_patch(patches.Rectangle((14, 44), 72, 8, fill=False, edgecolor='black', linewidth=0.9))
    ax.text(16, 49.2, "112: Gating Network & Routing Affinity Evaluator", fontweight='bold', fontsize=8.5)
    ax.text(16, 46, "- Evaluates softmax gating scores S_e(t); determines top-k expert assignments", fontsize=8)

    ax.add_patch(patches.Rectangle((14, 34), 72, 8, fill=False, edgecolor='black', linewidth=0.9))
    ax.text(16, 39.2, "114: Idempotent Capacity Projection Engine", fontweight='bold', fontsize=8.5)
    ax.text(16, 36, "- Enforces f(f(x)) = f(x); establishes invariant fixed points for retained tokens", fontsize=8)

    ax.add_patch(patches.Rectangle((14, 24), 72, 8, fill=False, edgecolor='black', linewidth=0.9))
    ax.text(16, 29.2, "116: 2D-Parallel In-Place Token Permutation Kernel", fontweight='bold', fontsize=8.5)
    ax.text(16, 26, "- O(1) auxiliary registers; executes zero-copy 2-cycle & multi-hop swaps across SMs", fontsize=8)

    # Arrow to Tensor Cores
    ax.annotate('', xy=(50, 23), xytext=(50, 18), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 120 Tensor Cores (GEMM)
    ax.add_patch(patches.Rectangle((10, 11), 80, 7, fill=False, edgecolor='black', linewidth=1.2))
    ax.text(50, 14.5, "120: PARALLEL EXPERT TENSOR CORES (Dense GEMM on Contiguous Active Tokens [0..C-1])", 
            ha='center', va='center', fontweight='bold', fontsize=8.5)

    ax.text(50, 4, "FIG. 1", ha='center', va='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    return fig


def create_fig2():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 96, "MIXTURE-OF-EXPERTS (MoE) TOKEN BUFFER STATE TRANSFORMATION", ha='center', fontweight='bold', fontsize=10.5)

    # STATE A
    ax.text(8, 88.5, "STATE A: Candidate Expert Token Buffer (Before Compaction)", fontweight='bold', fontsize=9)
    ax.add_patch(patches.Rectangle((8, 77), 84, 9, fill=False, edgecolor='black', linewidth=1.2))
    
    slots_A = [("0: Tok_0\n(High Score)", True), ("1: Tok_1\n(Low Score)", False), ("2: Tok_2\n(High Score)", True),
               ("3: Tok_3\n(Low Score)", False), ("4: Tok_4\n(High Score)", True), ("5: Tok_5\n(Low Score)", False)]
    for i, (txt, high) in enumerate(slots_A):
        x = 9 + i * 13.7
        style = 'solid' if high else 'dashed'
        ax.add_patch(patches.Rectangle((x, 78), 12.5, 7, fill=False, edgecolor='black', linestyle=style, linewidth=1))
        ax.text(x + 6.25, 81.5, txt, ha='center', va='center', fontsize=7.5)

    # MAPPING SECTION
    ax.text(8, 68, "IDEMPOTENT CAPACITY PROJECTION & DISJOINT CYCLE RESOLUTION", fontweight='bold', fontsize=9)
    ax.add_patch(patches.Rectangle((8, 42), 84, 23, fill=False, edgecolor='black', linewidth=1.2))
    ax.text(11, 62, "Idempotent Condition: f(f(x)) = f(x) over Expert Token Space", fontweight='bold', fontsize=8.5)
    ax.text(11, 58, "- Fixed Points (f(i) = i): Slots 0, 2 are high-affinity tokens already within capacity C.", fontsize=8)
    ax.text(11, 54, "- 2-Cycle Transposition Orbit (Slot 1 <-> Slot 4):", fontsize=8.5, fontweight='bold')
    ax.text(14, 50, "* Slot 1 (Low Score in Active Zone) swapped with Slot 4 (High Score in Tail Zone)", fontsize=8)
    ax.text(14, 46.5, "* O(1) In-Register Vectorized Exchange along Hidden Dimension D = 1024", fontsize=8)

    # STATE B
    ax.text(8, 33.5, "STATE B: Compacted Active Expert Buffer (After Capacity Truncation)", fontweight='bold', fontsize=9)
    ax.add_patch(patches.Rectangle((8, 18), 84, 12, fill=False, edgecolor='black', linewidth=1.2))

    slots_B = [("0: Tok_0\n(Retained)", True), ("1: Tok_4\n(Relocated)", True), ("2: Tok_2\n(Retained)", True),
               ("3: Tok_3\n(Dropped)", False), ("4: Tok_1\n(Dropped)", False), ("5: Tok_5\n(Dropped)", False)]
    for i, (txt, high) in enumerate(slots_B):
        x = 9 + i * 13.7
        style = 'solid' if high else 'dotted'
        ax.add_patch(patches.Rectangle((x, 21), 12.5, 7.5, fill=False, edgecolor='black', linestyle=style, linewidth=1))
        ax.text(x + 6.25, 24.75, txt, ha='center', va='center', fontsize=7.5)

    # Active Region Bracket
    ax.annotate('', xy=(9, 19), xytext=(50, 19), arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
    ax.text(29.5, 15, "Contiguous Active Expert Tokens (C=3)\nReady for High-Throughput GEMM on Tensor Cores", ha='center', fontsize=8, fontweight='bold')

    ax.text(50, 4, "FIG. 2", ha='center', va='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    return fig


def create_fig3():
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    ax.text(50, 96, "ZERO-COPY IN-PLACE MoE TOKEN DISPATCH AND COMPACTION ALGORITHM", ha='center', fontweight='bold', fontsize=10.5)

    steps = [
        ("302", "START: MoE Gating Network Computes Expert Routing Affinities S_e(t)", 89, 5),
        ("304", "Evaluate Expert Capacity Limit C = capacity_factor * (T * k / E)", 80, 5),
        ("306", "Construct Idempotent Projection Map f_e(x) Satisfying f(f(x)) = f(x)", 71, 5),
        ("308", "Initialize 2D Grid (pid_expert, pid_d_blk) across Streaming Multiprocessors", 62, 5),
        ("310", "Load Destination dest = f_e(i); Evaluate if dest > i", 53, 5),
        ("312", "Evaluate Transposition Fast-Path: Does f_e(dest) == i?", 44, 5),
        ("314", "Vectorized In-Register Token Swap along Hidden Dimension Slice (BLOCK_D=128)", 35, 5),
        ("316", "Generalized Cycle-Leader Traversal (curr < i Leader Verification)", 26, 5),
        ("318", "Update Active Buffer Boundary Pointer to Capacity C; Drop Excess Tail Tokens", 17, 5),
        ("320", "END: Zero Auxiliary VRAM Allocated; Continuous Tensor Core GEMM Ready", 8, 5),
    ]

    for code, text, y, h in steps:
        ax.add_patch(patches.Rectangle((12, y), 76, h, fill=False, edgecolor='black', linewidth=1.1))
        ax.text(14, y + h/2, code, fontweight='bold', va='center', fontsize=8)
        ax.text(20, y + h/2, text, va='center', fontsize=8)

    for i in range(len(steps) - 1):
        y_top = steps[i][2]
        y_bottom = steps[i+1][2] + steps[i+1][3]
        ax.annotate('', xy=(50, y_bottom), xytext=(50, y_top),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    ax.text(50, 2, "FIG. 3", ha='center', va='center', fontweight='bold', fontsize=12)
    plt.tight_layout()
    return fig


def main():
    patent_dir = "d:/ECETIN/ECETIN/studies/software/idempotent-permutations/moe-token-router/patent"
    paper_fig_dir = "d:/ECETIN/ECETIN/studies/software/idempotent-permutations/moe-token-router/paper/figures"
    os.makedirs(patent_dir, exist_ok=True)
    os.makedirs(paper_fig_dir, exist_ok=True)

    fig1 = create_fig1()
    fig1.savefig(os.path.join(patent_dir, "FIG_1_MoE_System_Architecture.pdf"))
    fig1.savefig(os.path.join(patent_dir, "FIG_1_MoE_System_Architecture.png"), dpi=300)
    fig1.savefig(os.path.join(paper_fig_dir, "FIG_1_MoE_System_Architecture.pdf"))

    fig2 = create_fig2()
    fig2.savefig(os.path.join(patent_dir, "FIG_2_MoE_Memory_State_Transition.pdf"))
    fig2.savefig(os.path.join(patent_dir, "FIG_2_MoE_Memory_State_Transition.png"), dpi=300)
    fig2.savefig(os.path.join(paper_fig_dir, "FIG_2_MoE_Memory_State_Transition.pdf"))

    fig3 = create_fig3()
    fig3.savefig(os.path.join(patent_dir, "FIG_3_MoE_Algorithm_Flowchart.pdf"))
    fig3.savefig(os.path.join(patent_dir, "FIG_3_MoE_Algorithm_Flowchart.png"), dpi=300)
    fig3.savefig(os.path.join(paper_fig_dir, "FIG_3_MoE_Algorithm_Flowchart.pdf"))

    pdf_multi = os.path.join(patent_dir, "Drawings_MoE_Token_Routing.pdf")
    with PdfPages(pdf_multi) as pdf:
        pdf.savefig(fig1)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    print("Successfully generated all patent and paper figures for MoE-Token-Router!")

if __name__ == "__main__":
    main()
