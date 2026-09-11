import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_uspto_provisional_spec():
    doc = Document()

    # --- Page Setup (USPTO 1-inch Margins) ---
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # --- Font Setup (Times New Roman, 12 pt) ---
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(12)
        p.paragraph_format.line_spacing = 1.5
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(13)
        return p

    def add_heading(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.5
        run = p.add_run(text)
        run.bold = True
        run.font.size = Pt(12)
        return p

    def add_para(num_str, text, is_claim=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 2.0  # USPTO Double-spaced standard
        if not is_claim:
            p.paragraph_format.first_line_indent = Inches(0.5)

        run_num = p.add_run(num_str + " ")
        run_num.bold = True
        p.add_run(text)
        return p

    # --- Title Block ---
    add_title("PATENT SPECIFICATION\nUNDER 35 U.S.C. § 111(b) (PROVISIONAL APPLICATION)")
    add_title("SYSTEM, METHOD, AND APPARATUS FOR ZERO-COPY IN-PLACE DYNAMIC TOKEN ROUTING AND CAPACITY COMPACTION IN SPARSE MIXTURE-OF-EXPERTS (MoE) ACCELERATORS")

    p_inv = doc.add_paragraph()
    p_inv.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_inv.paragraph_format.space_after = Pt(18)
    p_inv.paragraph_format.line_spacing = 1.5
    run_inv = p_inv.add_run("INVENTOR: A. Emre Çetin\nCitizenship: Republic of Turkey")
    run_inv.bold = True

    # --- Sections ---
    add_heading("CROSS-REFERENCE TO RELATED APPLICATIONS")
    add_para("[0001]", "This application claims the benefit of priority under 35 U.S.C. § 119(e) of provisional patent application filing parameters established on the filing date hereof.")

    add_heading("STATEMENT REGARDING FEDERALLY SPONSORED RESEARCH OR DEVELOPMENT")
    add_para("[0002]", "Not Applicable. No federal government funds or grants were used in the conception or development of this invention.")

    add_heading("FIELD OF THE INVENTION")
    add_para("[0003]", "The present invention relates generally to deep learning hardware accelerators, parallel processors, and memory management architectures. More specifically, the present invention relates to sparse Mixture-of-Experts (MoE) neural networks, dynamic token routing, token dropping and capacity compaction, zero-copy in-place memory permuting, and high-throughput Tensor Core matrix multiplication in Large Language Models (LLMs).")

    add_heading("BACKGROUND OF THE INVENTION")
    add_para("[0004]", "Sparse Mixture-of-Experts (MoE) architectures (e.g., Mixtral 8x7B, DeepSeek-V2/V3, Switch Transformer, GShard) achieve unprecedented parameter scaling while maintaining bounded computational cost per token. In an MoE layer, a sequence of T tokens is dynamically routed across E specialized expert feed-forward networks based on softmax affinity scores computed by a gating network.")
    add_para("[0005]", "To prevent workload imbalance and bound accelerator memory consumption, each expert enforces a strict computational capacity limit C = capacity_factor * (T * k / E). When the number of candidate tokens routed to an expert exceeds capacity C, excess tokens are discarded ('token dropping') or rerouted. However, current MoE computing platforms face severe hardware bottlenecks during token dispatch and compaction:")
    add_para("[0006]", "(1) High Auxiliary Allocation Overhead: Existing systems (e.g., Megatron-LM, DeepSpeed-MoE) allocate auxiliary global memory buffers of size O(E * C * D) via operating system calls (cudaMalloc) and execute out-of-place gather/scatter operations, inducing severe transient memory spikes and triggering Out-Of-Memory (OOM) faults during long-context inference; (2) Tensor Fragmentation and SIMD Memory Disalignment: Non-contiguous candidate tokens distributed across the sequence buffer prevent dense vector memory streaming into Tensor Cores; (3) High Kernel Dispatch Latency: Sorting and repacking candidate tokens sequentially induces high runtime overheads that degrade inference throughput.")
    add_para("[0007]", "Therefore, there exists a critical need in the art for an apparatus, system, and GPU kernel method capable of dynamically routing, compacting, and truncating candidate tokens directly within expert memory buffers in-place (zero-copy) with strictly bounded O(1) auxiliary scalar storage, guaranteed physical memory contiguity, and zero operating system memory allocation overhead.")

    add_heading("BRIEF SUMMARY OF THE INVENTION")
    add_para("[0008]", "The present invention solves the aforementioned technical deficiencies by providing a hardware-software co-designed architecture and high-throughput GPU execution kernel for zero-copy in-place dynamic token routing and capacity compaction in sparse Mixture-of-Experts accelerators.")
    add_para("[0009]", "In one embodiment, a hardware MoE routing controller evaluates gating scores across candidate tokens and constructs an idempotent projection map f_e: {0, ..., N-1} -> {0, ..., N-1} satisfying f_e(f_e(x)) = f_e(x). High-affinity tokens already located within the target capacity region [0, ..., C-1] form invariant fixed points (f_e(x) = x) that do not oscillate or incur data movement.")
    add_para("[0010]", "In another embodiment, the controller partitions candidate token positions into mutually disjoint permutation cycles. The kernel is mapped across a 2D accelerator execution grid (pid_expert, pid_d_blk), wherein disjoint tiles of the hidden embedding dimension D are permuted concurrently across Streaming Multiprocessors (SMs) using strictly O(1) scalar auxiliary registers without boolean marking bitmasks.")
    add_para("[0011]", "Transposition fast-paths (2-cycles) swap excess low-affinity tokens in the active zone with high-affinity tokens in the tail zone in a single in-register pipeline. Following compaction, an active boundary pointer is set to C, presenting a physically contiguous token buffer directly to Tensor Cores for dense General Matrix Multiplications (GEMM) with 100% elimination of peak auxiliary VRAM.")

    add_heading("BRIEF DESCRIPTION OF THE SEVERAL VIEWS OF THE DRAWINGS")
    add_para("[0012]", "FIG. 1 is a high-level system hardware architecture diagram illustrating a sparse Mixture-of-Experts accelerator comprising High Bandwidth Memory (HBM) storing candidate expert token buffers, a Hardware MoE Router and Token Consolidation Controller, and parallel Tensor Cores in accordance with an embodiment of the present invention.")
    add_para("[0013]", "FIG. 2 is a comparative memory state transition diagram showing physical token buffer layouts before and after idempotent in-place compaction, illustrating the transposition of tokens along disjoint permutation cycles into an active contiguous region and a dropped region.")
    add_para("[0014]", "FIG. 3 is a logical flowchart illustrating the execution steps of the zero-copy in-place MoE token dispatch and compaction algorithm executed with O(1) auxiliary register memory.")

    add_heading("DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS")
    add_para("[0015]", "Referring to FIG. 1, a sparse Mixture-of-Experts (MoE) accelerator (100) (e.g., GPU, NPU, or domain-specific neural processor) includes a primary accelerator memory (102) storing candidate expert token embedding buffers (104) having shape [E, N, D]. An integrated Hardware MoE Router and Token Consolidation Controller (110) couples the memory (102) to Parallel Expert Tensor Cores (120).")
    add_para("[0016]", "The Controller (110) includes: (1) A Gating Network and Routing Affinity Evaluator (112) configured to compute softmax gating scores S_e(t); (2) An Idempotent Capacity Projection Engine (114) configured to establish fixed points and cyclic orbits satisfying f_e(f_e(x)) = f_e(x); and (3) A 2D-Parallel In-Place Token Permutation Kernel (116) configured to execute zero-copy token swaps across hidden dimension tiles using O(1) scalar auxiliary registers.")
    add_para("[0017]", "Referring to FIG. 2, State A depicts an initial candidate token buffer where high-affinity tokens (Tok_0, Tok_2, Tok_4) are interspersed with low-affinity tokens (Tok_1, Tok_3, Tok_5). The idempotent mapping identifies Tok_0 and Tok_2 as invariant fixed points. A 2-cycle transposition orbit is formed between Tok_1 in the active zone and Tok_4 in the tail zone. Direct register exchanges swap the embedding vectors between slots 1 and 4 across hidden dimension D. State B shows the resulting compacted buffer where slots [0, 1, 2] form a strictly contiguous memory layout ready for Tensor Core GEMM.")
    add_para("[0018]", "Referring to FIG. 3, at step 302 token routing begins. At step 304, capacity limit C is evaluated. At step 306, idempotent map f_e is generated. At step 308, the 2D execution grid is launched. Steps 310-312 check destination indices and transposition fast-paths. Step 314 executes vectorized in-register token swaps. Step 316 handles generalized cycle-leader search. At step 318, the active boundary pointer is truncated to capacity C, delivering 0.00 MB auxiliary VRAM compaction at step 320.")

    add_heading("CLAIMS")
    p_claim_intro = doc.add_paragraph()
    p_claim_intro.paragraph_format.line_spacing = 2.0
    p_claim_intro.add_run("What is claimed is:").bold = True

    add_para("[0019]", "Claim 1. A computer-implemented method for zero-copy in-place dynamic token routing and capacity compaction in a sparse Mixture-of-Experts (MoE) neural network accelerator, the method comprising: (a) evaluating routing affinity scores across N candidate tokens assigned to an expert buffer within primary accelerator memory; (b) generating an idempotent index projection map f(i) satisfying f(f(x)) = f(x) over an expert capacity threshold C, establishing invariant fixed points for retained tokens; (c) identifying a set of cycle leaders corresponding to mutually disjoint permutation cycles using strictly bounded O(1) scalar auxiliary register storage without external boolean marking bitmasks; (d) executing an in-place cyclic permutation of multi-dimensional token embeddings along each disjoint permutation cycle using temporary register storage, shifting token payloads directly into contiguous destination addresses; and (e) truncating an active buffer boundary pointer to capacity C to present a contiguous block of active tokens for dense matrix multiplication while dropping excess tokens without issuing host operating system memory allocation commands.", is_claim=True)

    add_para("[0020]", "Claim 2. A sparse Mixture-of-Experts computing system comprising: a primary accelerator memory storing a plurality of candidate expert token embeddings within a contiguous buffer; an execution engine comprising a plurality of parallel tensor cores; and an integrated memory management controller operatively coupled to the memory and execution engine, configured to: compute gating routing scores for candidate tokens; construct an idempotent permutation map translating routing scores into contiguous target index addresses within an expert capacity limit C; dispatch a 2D-parallel in-place token compaction kernel to permute token embeddings along disjoint permutation cycles via cycle-leader validation using strictly bounded O(1) auxiliary register memory; and output a physically contiguous, compacted token sequence without auxiliary memory buffer allocations.", is_claim=True)

    add_para("[0021]", "Claim 3. The system of Claim 2, wherein the 2D-parallel in-place token compaction kernel comprises a 2-cycle transposition fast-path that detects reciprocal index pairings f(f(i)) = i where f(i) > i and executes direct in-register vectorized exchanges across embedding tiles of width BLOCK_D in a single execution pipeline.", is_claim=True)

    add_para("[0022]", "Claim 4. The system of Claim 2, wherein the 2D-parallel in-place token compaction kernel is partitioned across a 2D accelerator execution grid (pid_expert, pid_d_blk), wherein pid_expert indexes independent expert candidate buffers and pid_d_blk indexes disjoint tiles of token hidden dimension D, executing in-place token compaction concurrently across multiple Streaming Multiprocessors with zero inter-expert memory locking.", is_claim=True)

    patent_dir = "d:/ECETIN/ECETIN/studies/software/idempotent-permutations/moe-token-router/patent"
    filename = os.path.join(patent_dir, "USPTO_Provisional_Specification_MoE_Router_Cetin.docx")
    doc.save(filename)
    print(f"Patent specification successfully generated: {filename}")

if __name__ == "__main__":
    create_uspto_provisional_spec()
