#!/usr/bin/env python3
# A complete Manim script for explaining the Epigenomic Transformer

# First import manim
from manim import *

# Then import other libraries
import numpy as np
import os
import sys

# Set a consistent color scheme
METHYLATION_COLOR = "#3B4CC0"  # Blue for methylation
GENE_COLOR = "#B83A4D"  # Red for genes
TRANSFORMER_COLOR = "#38761D"  # Green for transformer components
ATTENTION_COLOR = "#F1C232"  # Yellow for attention
EXPERT_COLOR = "#8E7CC3"  # Purple for experts
BACKGROUND_COLOR = "#1E1E1E"  # Dark background

# Now config is available because it came from the manim import
config.background_color = BACKGROUND_COLOR
config.pixel_height = 1080
config.pixel_width = 1920
config.frame_height = 8
config.frame_width = 14.2

class Introduction(Scene):
    """Introduction to the Epigenomic Transformer concept."""
    
    def construct(self):
        # Title sequence
        title = Text("Epigenomic Transformers", font_size=72, color=WHITE)
        subtitle = Text("Mathematical Models for Disease Classification", font_size=42, color=WHITE)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title, run_time=2))
        self.play(FadeIn(subtitle, shift=UP*0.5))
        self.wait(2)
        
        self.play(
            title.animate.scale(0.7).to_edge(UP),
            FadeOut(subtitle)
        )
        self.wait()
        
        # DNA visualization
        dna_helix = self.create_dna_helix()
        dna_helix.shift(LEFT * 3)
        
        self.play(Create(dna_helix, run_time=3))
        self.wait()
        
        # Introduce epigenetics concept
        epigenetics = Text("Epigenetics:", font_size=48, color=WHITE)
        epigenetics.to_edge(LEFT).shift(UP * 2)
        
        definition = Text(
            "How cells control gene expression\nwithout changing DNA sequence", 
            font_size=36, color=WHITE
        )
        definition.next_to(epigenetics, DOWN, aligned_edge=LEFT, buff=0.5)
        
        self.play(Write(epigenetics))
        self.play(Write(definition, run_time=2))
        self.wait()
        
        # Show methylation
        methylation_text = Text("DNA Methylation", font_size=42, color=METHYLATION_COLOR)
        methylation_text.next_to(definition, DOWN, aligned_edge=LEFT, buff=1)
        
        # Create methyl groups on the DNA
        methyl_groups = self.add_methyl_groups(dna_helix)
        
        self.play(Write(methylation_text))
        self.play(
            *[GrowFromCenter(methyl) for methyl in methyl_groups],
            run_time=2
        )
        self.wait()
        
        # Explain ME/CFS and Long COVID
        disease_text = Text("ME/CFS & Long COVID", font_size=42, color=GENE_COLOR)
        disease_text.to_edge(RIGHT).shift(UP * 2)
        
        disease_desc = Text(
            "Post-viral conditions with\nepigenetic alterations",
            font_size=36, color=WHITE
        )
        disease_desc.next_to(disease_text, DOWN, aligned_edge=LEFT, buff=0.5)
        
        self.play(Write(disease_text))
        self.play(Write(disease_desc, run_time=2))
        self.wait(2)
        
        # Transition to the mathematical challenge
        challenge = Text("The Mathematical Challenge", font_size=58, color=WHITE)
        challenge.center()
        
        self.play(
            FadeOut(dna_helix),
            FadeOut(methyl_groups),
            FadeOut(epigenetics),
            FadeOut(definition),
            FadeOut(methylation_text),
            FadeOut(disease_text),
            FadeOut(disease_desc),
            FadeOut(title),
            run_time=1.5
        )
        
        self.play(Write(challenge))
        self.wait(2)
        self.play(FadeOut(challenge))
    
    def create_dna_helix(self):
        """Create a stylized DNA double helix."""
        # Parameters for the helix
        t_range = np.linspace(0, 4*np.pi, 100)
        radius = 0.5
        frequency = 1
        
        # Create the two strands
        strand1_points = []
        strand2_points = []
        
        for t in t_range:
            x1 = radius * np.cos(frequency * t)
            y1 = t / 3  # Stretch along y-axis
            strand1_points.append([x1, y1, 0])
            
            x2 = radius * np.cos(frequency * t + np.pi)
            y2 = t / 3
            strand2_points.append([x2, y2, 0])
        
        # Create the strands as polygons
        strand1 = Polygon(*strand1_points, color=BLUE, stroke_width=3)
        strand2 = Polygon(*strand2_points, color=RED, stroke_width=3)
        
        # Create the base pairs (rungs)
        rungs = VGroup()
        for i in range(0, len(t_range), 10):
            p1 = strand1_points[i]
            p2 = strand2_points[i]
            line = Line(p1, p2, color=WHITE, stroke_width=2)
            rungs.add(line)
        
        return VGroup(strand1, strand2, rungs)
    
    def add_methyl_groups(self, dna_helix):
        """Add methyl groups to the DNA."""
        methyl_groups = VGroup()
        strand1 = dna_helix[0]
        
        # Add methyl groups at selected positions
        positions = [10, 30, 50, 70, 90]
        for i in positions:
            point = strand1.points[i]
            methyl = Circle(radius=0.15, color=METHYLATION_COLOR, fill_opacity=0.7)
            methyl.move_to(point)
            methyl_groups.add(methyl)
        
        return methyl_groups

class DataRepresentation(Scene):
    """Explaining how methylation data is represented mathematically."""
    
    def construct(self):
        # Title
        title = Text("Mathematical Representation of Methylation Data", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Explain what a CpG site is
        cpg = Text("CpG Site: Where cytosine can be methylated", font_size=36)
        cpg.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(cpg))
        self.wait()
        
        # Show the data format
        matrix_title = Text("Methylation Data Matrix:", font_size=36)
        matrix_title.next_to(cpg, DOWN, buff=0.8).to_edge(LEFT)
        
        self.play(Write(matrix_title))
        
        # Create a sample matrix of methylation values
        sample_matrix = self.create_methylation_matrix()
        sample_matrix.next_to(matrix_title, RIGHT, buff=0.5)
        
        self.play(Create(sample_matrix))
        self.wait()
        
        # Explain beta values
        beta_text = Text("β-values: 0 (unmethylated) to 1 (fully methylated)", font_size=32)
        beta_text.next_to(matrix_title, DOWN, aligned_edge=LEFT, buff=0.5)
        
        self.play(Write(beta_text))
        self.wait()
        
        # Highlight different patterns
        self.highlight_patterns(sample_matrix)
        
        # Show the mathematical challenge of dimensionality
        challenge = Text("Challenge: 1,280 CpG sites × 70 patients", font_size=36)
        challenge.next_to(beta_text, DOWN, buff=1)
        
        self.play(Write(challenge))
        self.wait()
        
        # Transition to mathematical embedding
        embedding_text = Text("We need to embed this high-dimensional data", font_size=36)
        embedding_text.next_to(challenge, DOWN, buff=0.5)
        
        self.play(Write(embedding_text))
        self.wait(2)
        
        # Transition to next scene
        self.play(
            FadeOut(title),
            FadeOut(cpg),
            FadeOut(matrix_title),
            FadeOut(sample_matrix),
            FadeOut(beta_text),
            FadeOut(challenge),
            FadeOut(embedding_text),
            run_time=1.5
        )
    
    def create_methylation_matrix(self):
        """Create a visualization of methylation data matrix."""
        # Create a 10x16 grid of squares to represent methylation values
        n_rows, n_cols = 5, 8
        square_size = 0.4
        
        grid = VGroup()
        values = np.random.rand(n_rows, n_cols)  # Random values between 0 and 1
        
        for i in range(n_rows):
            for j in range(n_cols):
                value = values[i, j]
                color = self.get_methylation_color(value)
                square = Square(side_length=square_size)
                square.set_fill(color, opacity=0.8)
                square.set_stroke(WHITE, width=1)
                square.move_to([j*square_size, -i*square_size, 0])
                grid.add(square)
                
                # Add value text
                value_text = Text(f"{value:.2f}", font_size=14)
                value_text.move_to(square.get_center())
                grid.add(value_text)
        
        # Add row and column labels
        row_labels = VGroup()
        for i in range(n_rows):
            label = Text(f"Patient {i+1}", font_size=18)
            label.next_to(grid[i*2*n_cols], LEFT, buff=0.3)
            row_labels.add(label)
        
        col_labels = VGroup()
        for j in range(n_cols):
            label = Text(f"CpG {j+1}", font_size=18)
            label.next_to(grid[j*2], UP, buff=0.3)
            col_labels.add(label)
        
        return VGroup(grid, row_labels, col_labels)
    
    def get_methylation_color(self, value):
        """Map methylation value to color (blue for low, red for high)."""
        r = value
        g = 0
        b = 1 - value
        return rgb_to_color([r, g, b])
    
    def highlight_patterns(self, matrix):
        """Highlight different methylation patterns in the matrix."""
        grid = matrix[0]
        
        # Highlight a column (CpG site across patients)
        col_highlight = SurroundingRectangle(
            VGroup(*[grid[i*16 + 4] for i in range(5)]), 
            color=YELLOW, 
            buff=0.05
        )
        col_text = Text("CpG site pattern", font_size=24, color=YELLOW)
        col_text.next_to(col_highlight, RIGHT)
        
        self.play(Create(col_highlight), Write(col_text))
        self.wait()
        
        # Highlight a row (patient's methylation profile)
        row_highlight = SurroundingRectangle(
            VGroup(*[grid[2*16 + j] for j in range(8)]), 
            color=GREEN, 
            buff=0.05
        )
        row_text = Text("Patient's methylation profile", font_size=24, color=GREEN)
        row_text.next_to(row_highlight, RIGHT)
        
        self.play(Create(row_highlight), Write(row_text))
        self.wait(2)
        
        self.play(
            FadeOut(col_highlight),
            FadeOut(col_text),
            FadeOut(row_highlight),
            FadeOut(row_text)
        )

class TransformerOverview(Scene):
    """Overview of the Transformer architecture."""
    
    def construct(self):
        # Title
        title = Text("The Transformer Architecture", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Explain why transformers are suitable
        explanation = Text(
            "Transformers excel at modeling complex patterns\nand interactions between features",
            font_size=36,
            line_spacing=1.2
        )
        explanation.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(explanation, run_time=2))
        self.wait()
        
        # Show the overall transformer block diagram
        transformer_diagram = self.create_transformer_diagram()
        transformer_diagram.scale(0.9).next_to(explanation, DOWN, buff=1)
        
        self.play(Create(transformer_diagram, run_time=3))
        self.wait(2)
        
        # Explain key innovations for epigenomic data
        self.play(
            FadeOut(explanation),
            transformer_diagram.animate.scale(0.8).to_edge(LEFT)
        )
        
        innovations = VGroup(
            Text("Key Innovations:", font_size=40, color=YELLOW),
            Text("1. Self-supervised Masked Pretraining", font_size=32),
            Text("2. Mixture-of-Experts Feed-Forward Network", font_size=32),
            Text("3. Adaptive Computation Time", font_size=32)
        )
        
        innovations.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        innovations.next_to(title, DOWN, buff=0.8).to_edge(RIGHT)
        
        for i in range(len(innovations)):
            self.play(FadeIn(innovations[i], shift=UP*0.2))
            self.wait(0.5)
        
        self.wait(2)
        
        # Transition to detailed explanation
        next_text = Text("Let's explore each component in detail", font_size=36)
        next_text.to_edge(DOWN, buff=1)
        
        self.play(Write(next_text))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(transformer_diagram),
            FadeOut(innovations),
            FadeOut(next_text),
            run_time=1.5
        )
    
    def create_transformer_diagram(self):
        """Create a simplified diagram of the transformer architecture."""
        # Main blocks
        input_block = Rectangle(width=4, height=0.8, color=WHITE)
        input_text = Text("Input Methylation Data", font_size=24)
        input_text.move_to(input_block.get_center())
        input_group = VGroup(input_block, input_text)
        
        embedding_block = Rectangle(width=4, height=0.8, color=BLUE)
        embedding_text = Text("Input Embedding", font_size=24)
        embedding_text.move_to(embedding_block.get_center())
        embedding_group = VGroup(embedding_block, embedding_text)
        embedding_group.next_to(input_group, DOWN, buff=0.5)
        
        # Transformer encoder block
        encoder_block = Rectangle(width=4, height=4, color=TRANSFORMER_COLOR)
        encoder_block.next_to(embedding_group, DOWN, buff=0.5)
        
        attn_block = Rectangle(width=3.5, height=1.2, color=ATTENTION_COLOR)
        attn_text = Text("Multi-Head Self-Attention", font_size=20)
        attn_text.move_to(attn_block.get_center())
        attn_group = VGroup(attn_block, attn_text)
        attn_group.move_to(encoder_block.get_center() + UP * 1)
        
        ffn_block = Rectangle(width=3.5, height=1.2, color=EXPERT_COLOR)
        ffn_text = Text("MoE Feed-Forward", font_size=20)
        ffn_text.move_to(ffn_block.get_center())
        ffn_group = VGroup(ffn_block, ffn_text)
        ffn_group.move_to(encoder_block.get_center() + DOWN * 1)
        
        encoder_text = Text("Transformer Encoder", font_size=24, color=TRANSFORMER_COLOR)
        encoder_text.next_to(encoder_block, LEFT, buff=0.3)
        
        # ACT mechanism
        act_block = Rectangle(width=4, height=0.8, color=RED)
        act_text = Text("Adaptive Computation Time", font_size=20)
        act_text.move_to(act_block.get_center())
        act_group = VGroup(act_block, act_text)
        act_group.next_to(encoder_block, DOWN, buff=0.5)
        
        # Final classification block
        output_block = Rectangle(width=4, height=0.8, color=GREEN)
        output_text = Text("Classification Output", font_size=24)
        output_text.move_to(output_block.get_center())
        output_group = VGroup(output_block, output_text)
        output_group.next_to(act_group, DOWN, buff=0.5)
        
        # Arrows
        arrows = VGroup(
            Arrow(input_group.get_bottom(), embedding_group.get_top()),
            Arrow(embedding_group.get_bottom(), encoder_block.get_top()),
            Arrow(encoder_block.get_bottom(), act_group.get_top()),
            Arrow(act_group.get_bottom(), output_group.get_top())
        )
        
        # x6 label for encoder blocks
        x6_text = Text("×6", font_size=28, color=TRANSFORMER_COLOR)
        x6_text.next_to(encoder_block, RIGHT, buff=0.3)
        
        # Create a loop arrow from ACT back to encoder
        loop_arrow = CurvedArrow(
            act_group.get_left() + LEFT * 0.2,
            encoder_block.get_left() + LEFT * 0.2,
            angle=-np.pi/2
        )
        loop_text = Text("If needed", font_size=18, color=RED)
        loop_text.next_to(loop_arrow, LEFT)
        
        return VGroup(
            input_group, embedding_group, encoder_block, attn_group, ffn_group,
            act_group, output_group, arrows, encoder_text, x6_text, loop_arrow, loop_text
        )

class InputEmbedding(Scene):
    """Explaining how methylation data is embedded for the transformer."""
    
    def construct(self):
        # Title
        title = Text("Input Embedding for Methylation Data", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Showing the methylation vector
        vector_title = Text("Original Methylation Values:", font_size=36)
        vector_title.next_to(title, DOWN, buff=0.8).to_edge(LEFT)
        
        self.play(Write(vector_title))
        
        # Create a visualization of the methylation vector
        values = np.random.rand(10)  # 10 example values
        meth_vector = self.create_feature_vector(values, "x")
        meth_vector.next_to(vector_title, DOWN, buff=0.4, aligned_edge=LEFT)
        
        self.play(Create(meth_vector))
        self.wait()
        
        # Show the embedding process
        embed_arrow = Arrow(
            meth_vector.get_right() + RIGHT * 0.5,
            meth_vector.get_right() + RIGHT * 2,
            buff=0
        )
        embed_text = MathTex(r"\mathbf{e} = W_e \mathbf{x} + \mathbf{b}_e", font_size=36)
        embed_text.next_to(embed_arrow, UP, buff=0.2)
        
        self.play(GrowArrow(embed_arrow), Write(embed_text))
        self.wait()
        
        # Show the embedded vector
        embedded_values = np.random.rand(10)  # Example embedded values
        embedded_vector = self.create_feature_vector(embedded_values, "e")
        embedded_vector.next_to(embed_arrow, RIGHT, buff=0.5)
        
        self.play(Create(embedded_vector))
        self.wait()
        
        # Explain chunking process
        self.play(
            meth_vector.animate.shift(UP * 1.5),
            embed_arrow.animate.shift(UP * 1.5),
            embed_text.animate.shift(UP * 1.5),
            embedded_vector.animate.shift(UP * 1.5)
        )
        
        chunking_title = Text("Chunking into L=4 Tokens:", font_size=36)
        chunking_title.next_to(meth_vector, DOWN, buff=1, aligned_edge=LEFT)
        
        self.play(Write(chunking_title))
        
        # Visualize the chunking
        chunks = self.create_chunks(embedded_values)
        chunks.next_to(chunking_title, DOWN, buff=0.5, aligned_edge=LEFT)
        
        self.play(Create(chunks))
        self.wait()
        
        # Show positional encoding
        pos_enc = Text("+ Positional Encoding", font_size=32)
        pos_enc.next_to(chunks, RIGHT, buff=1)
        
        pos_arrow = Arrow(chunks.get_right(), pos_enc.get_left(), buff=0.2)
        
        self.play(GrowArrow(pos_arrow), Write(pos_enc))
        self.wait()
        
        # Final form before transformer
        final_form = MathTex(r"Z^0 = E + P", font_size=42)
        final_form.next_to(chunks, DOWN, buff=1)
        
        self.play(Write(final_form))
        self.wait(2)
        
        # Summary of what we've done
        summary = Text(
            "We've transformed methylation data into a format\nsuitable for transformer processing",
            font_size=36,
            line_spacing=1.2
        )
        summary.to_edge(DOWN, buff=1)
        
        self.play(Write(summary))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(vector_title),
            FadeOut(meth_vector),
            FadeOut(embed_arrow),
            FadeOut(embed_text),
            FadeOut(embedded_vector),
            FadeOut(chunking_title),
            FadeOut(chunks),
            FadeOut(pos_enc),
            FadeOut(pos_arrow),
            FadeOut(final_form),
            FadeOut(summary),
            run_time=1.5
        )
    
    def create_feature_vector(self, values, symbol):
        """Create a visualization of a feature vector."""
        n_values = len(values)
        square_size = 0.4
        
        vector = VGroup()
        
        # Create squares with values
        for i, val in enumerate(values):
            color = self.get_value_color(val)
            square = Square(side_length=square_size)
            square.set_fill(color, opacity=0.8)
            square.set_stroke(WHITE, width=1)
            square.move_to([i*square_size, 0, 0])
            
            # Add value text
            value_text = Text(f"{val:.2f}", font_size=14)
            value_text.move_to(square.get_center())
            
            vector.add(VGroup(square, value_text))
        
        # Add brackets and labels
        left_bracket = Text("[", font_size=36)
        left_bracket.next_to(vector[0], LEFT, buff=0.1)
        
        right_bracket = Text("]", font_size=36)
        right_bracket.next_to(vector[-1], RIGHT, buff=0.1)
        
        vector_label = MathTex(f"\\mathbf{{{symbol}}}", font_size=36)
        vector_label.next_to(left_bracket, LEFT, buff=0.2)
        
        return VGroup(vector, left_bracket, right_bracket, vector_label)
    
    def create_chunks(self, values):
        """Create a visualization of chunked tokens."""
        n_chunks = 4
        chunk_size = len(values) // n_chunks
        square_size = 0.4
        
        chunks = VGroup()
        
        for c in range(n_chunks):
            chunk = VGroup()
            for i in range(chunk_size):
                idx = c * chunk_size + i
                if idx < len(values):
                    val = values[idx]
                    color = self.get_value_color(val)
                    square = Square(side_length=square_size)
                    square.set_fill(color, opacity=0.8)
                    square.set_stroke(WHITE, width=1)
                    square.move_to([i*square_size, 0, 0])
                    chunk.add(square)
            
            # Add bracket and label
            chunk_label = Text(f"Token {c+1}", font_size=24)
            chunk_label.next_to(chunk, DOWN, buff=0.2)
            
            chunk_group = VGroup(chunk, chunk_label)
            if c > 0:
                chunk_group.next_to(chunks[-1], RIGHT, buff=0.5)
            
            chunks.add(chunk_group)
        
        return chunks
    
    def get_value_color(self, value):
        """Map value to color (blue to red gradient)."""
        r = value
        g = 0.2
        b = 1 - value
        return rgb_to_color([r, g, b])

class SelfAttentionMechanism(Scene):
    """Explaining the self-attention mechanism in detail."""
    
    def construct(self):
        # Title
        title = Text("Self-Attention Mechanism", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Explain the key insight
        insight = Text(
            "Key insight: Let the model learn which features\nshould pay attention to which other features",
            font_size=36,
            line_spacing=1.2
        )
        insight.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(insight, run_time=2))
        self.wait(2)
        
        # Transition to detailed explanation
        self.play(
            insight.animate.scale(0.8).to_edge(UP, buff=1.5),
            title.animate.shift(UP * 0.5)
        )
        
        # Explain the Query, Key, Value concept
        qkv_title = Text("Query, Key, Value Projections:", font_size=36)
        qkv_title.next_to(insight, DOWN, buff=0.8)
        
        self.play(Write(qkv_title))
        
        # Create visualization for QKV
        qkv_diagram = self.create_qkv_diagram()
        qkv_diagram.next_to(qkv_title, DOWN, buff=0.5)
        
        self.play(Create(qkv_diagram, run_time=3))
        self.wait()
        
        # Show the mathematical formula
        self.play(
            FadeOut(qkv_title),
            qkv_diagram.animate.scale(0.8).to_edge(LEFT)
        )
        
        formulas = VGroup(
            MathTex(r"Q = Z^{l-1}W_Q", font_size=36),
            MathTex(r"K = Z^{l-1}W_K", font_size=36),
            MathTex(r"V = Z^{l-1}W_V", font_size=36)
        )
        
        formulas.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        formulas.next_to(insight, DOWN, buff=0.8).to_edge(RIGHT)
        
        self.play(Write(formulas[0]))
        self.wait(0.5)
        self.play(Write(formulas[1]))
        self.wait(0.5)
        self.play(Write(formulas[2]))
        self.wait()
        
        # Transition to attention score calculation
        self.play(
            FadeOut(qkv_diagram),
            FadeOut(formulas)
        )
        
        # Show attention score calculation
        attn_title = Text("Calculating Attention Scores:", font_size=36)
        attn_title.next_to(insight, DOWN, buff=0.8).to_edge(LEFT)
        
        self.play(Write(attn_title))
        
        # Show matrix multiplication
        matrix_mult = self.create_matrix_multiplication_visual()
        matrix_mult.next_to(attn_title, DOWN, buff=0.5)
        
        self.play(Create(matrix_mult, run_time=2))
        self.wait()
        
        # Show scaling and softmax
        scaling_text = MathTex(r"\text{Scaled Attention} = \frac{QK^T}{\sqrt{d}}", font_size=36)
        scaling_text.next_to(matrix_mult, DOWN, buff=0.8)
        
        self.play(Write(scaling_text))
        self.wait()
        
        softmax_text = MathTex(r"A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)", font_size=36)
        softmax_text.next_to(scaling_text, DOWN, buff=0.5)
        
        self.play(Write(softmax_text))
        self.wait()
        
        # Visualize softmax transformation
        softmax_viz = self.create_softmax_visualization()
        softmax_viz.next_to(softmax_text, DOWN, buff=0.5)
        
        self.play(Create(softmax_viz))
        self.wait(2)
        
        # Show final attention output calculation
        self.play(
            FadeOut(matrix_mult),
            FadeOut(scaling_text),
            FadeOut(softmax_text),
            FadeOut(softmax_viz)
        )
        
        # Final attention output
        output_title = Text("Final Attention Output:", font_size=36)
        output_title.next_to(attn_title, DOWN, buff=0.8)
        
        self.play(Write(output_title))
        
        output_formula = MathTex(r"\text{Attention}(Q,K,V) = AV", font_size=36)
        output_formula.next_to(output_title, DOWN, buff=0.5)
        
        self.play(Write(output_formula))
        self.wait()
        
        # Multi-head attention
        multihead_title = Text("Multi-Head Attention:", font_size=36)
        multihead_title.next_to(output_formula, DOWN, buff=0.8)
        
        self.play(Write(multihead_title))
        
        multihead_formula = MathTex(r"\text{MultiHead}(Z) = \text{Concat}(\text{head}_1,...,\text{head}_8)W_O", font_size=32)
        multihead_formula.next_to(multihead_title, DOWN, buff=0.5)
        
        self.play(Write(multihead_formula))
        
        # Explain multi-head concept
        multihead_explanation = Text(
            "8 separate attention mechanisms run in parallel\neach focusing on different patterns",
            font_size=28,
            line_spacing=1.2
        )
        multihead_explanation.next_to(multihead_formula, DOWN, buff=0.5)
        
        self.play(Write(multihead_explanation))
        self.wait(2)
        
        # Explanation of biological significance
        biological_box = SurroundingRectangle(multihead_explanation, buff=0.5, color=YELLOW)
        biological_text = Text(
            "This allows the model to focus on different biological pathways:\nimmune genes, stress response genes, etc.",
            font_size=28,
            color=YELLOW,
            line_spacing=1.2
        )
        biological_text.next_to(biological_box, DOWN, buff=0.5)
        
        self.play(Create(biological_box), Write(biological_text))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(insight),
            FadeOut(attn_title),
            FadeOut(output_title),
            FadeOut(output_formula),
            FadeOut(multihead_title),
            FadeOut(multihead_formula),
            FadeOut(multihead_explanation),
            FadeOut(biological_box),
            FadeOut(biological_text),
            run_time=1.5
        )
    
    def create_qkv_diagram(self):
        """Create a visualization of the Query, Key, Value concept."""
        # Input tensor
        input_rect = Rectangle(width=4, height=0.8, color=WHITE)
        input_text = Text("Input Z", font_size=24)
        input_text.move_to(input_rect.get_center())
        input_group = VGroup(input_rect, input_text)
        
        # Projection matrices
        wq_rect = Rectangle(width=1.2, height=1.2, color=RED)
        wq_text = Text("WQ", font_size=24)
        wq_text.move_to(wq_rect.get_center())
        wq_group = VGroup(wq_rect, wq_text)
        wq_group.next_to(input_group, DOWN+RIGHT, buff=1)
        
        wk_rect = Rectangle(width=1.2, height=1.2, color=GREEN)
        wk_text = Text("WK", font_size=24)
        wk_text.move_to(wk_rect.get_center())
        wk_group = VGroup(wk_rect, wk_text)
        wk_group.next_to(input_group, DOWN, buff=1)
        
        wv_rect = Rectangle(width=1.2, height=1.2, color=BLUE)
        wv_text = Text("WV", font_size=24)
        wv_text.move_to(wv_rect.get_center())
        wv_group = VGroup(wv_rect, wv_text)
        wv_group.next_to(input_group, DOWN+LEFT, buff=1)
        
        # Output tensors
        q_rect = Rectangle(width=1.2, height=0.8, color=RED)
        q_text = Text("Q", font_size=24)
        q_text.move_to(q_rect.get_center())
        q_group = VGroup(q_rect, q_text)
        q_group.next_to(wq_group, DOWN, buff=0.7)
        
        k_rect = Rectangle(width=1.2, height=0.8, color=GREEN)
        k_text = Text("K", font_size=24)
        k_text.move_to(k_rect.get_center())
        k_group = VGroup(k_rect, k_text)
        k_group.next_to(wk_group, DOWN, buff=0.7)
        
        v_rect = Rectangle(width=1.2, height=0.8, color=BLUE)
        v_text = Text("V", font_size=24)
        v_text.move_to(v_rect.get_center())
        v_group = VGroup(v_rect, v_text)
        v_group.next_to(wv_group, DOWN, buff=0.7)
        
        # Arrows
        arrows = VGroup(
            Arrow(input_group.get_bottom(), wq_group.get_top(), buff=0.1),
            Arrow(input_group.get_bottom(), wk_group.get_top(), buff=0.1),
            Arrow(input_group.get_bottom(), wv_group.get_top(), buff=0.1),
            Arrow(wq_group.get_bottom(), q_group.get_top(), buff=0.1),
            Arrow(wk_group.get_bottom(), k_group.get_top(), buff=0.1),
            Arrow(wv_group.get_bottom(), v_group.get_top(), buff=0.1)
        )
        
        # Labels
        query_label = Text("Query: 'What am I looking for?'", font_size=20, color=RED)
        query_label.next_to(q_group, RIGHT, buff=0.5)
        
        key_label = Text("Key: 'What do I contain?'", font_size=20, color=GREEN)
        key_label.next_to(k_group, RIGHT, buff=0.5)
        
        value_label = Text("Value: 'What info do I carry?'", font_size=20, color=BLUE)
        value_label.next_to(v_group, RIGHT, buff=0.5)
        
        return VGroup(
            input_group, wq_group, wk_group, wv_group, q_group, k_group, v_group,
            arrows, query_label, key_label, value_label
        )
    
    def create_matrix_multiplication_visual(self):
        """Create a visualization of QK^T matrix multiplication."""
        # Q and K matrices
        q_matrix = self.create_matrix("Q", 4, 3, RED)
        k_matrix = self.create_matrix("K", 4, 3, GREEN)
        k_transpose = self.create_matrix("K^T", 3, 4, GREEN)
        
        # Position matrices
        q_matrix.to_edge(LEFT, buff=1)
        k_transpose.next_to(q_matrix, RIGHT, buff=0.5)
        
        # Multiplication operator
        mult_symbol = MathTex(r"\times", font_size=36)
        mult_symbol.move_to((q_matrix.get_right() + k_transpose.get_left()) / 2)
        
        # Result matrix
        result_matrix = self.create_matrix("QK^T", 4, 4, YELLOW)
        result_matrix.next_to(k_transpose, RIGHT, buff=1)
        
        # Equal sign
        equal_sign = MathTex(r"=", font_size=36)
        equal_sign.move_to((k_transpose.get_right() + result_matrix.get_left()) / 2)
        
        # Show matrix dimensions
        q_dim = Text("(4×3)", font_size=20, color=RED)
        q_dim.next_to(q_matrix, DOWN, buff=0.2)
        
        k_dim = Text("(3×4)", font_size=20, color=GREEN)
        k_dim.next_to(k_transpose, DOWN, buff=0.2)
        
        result_dim = Text("(4×4)", font_size=20, color=YELLOW)
        result_dim.next_to(result_matrix, DOWN, buff=0.2)
        
        # Explanation
        explanation = Text(
            "QK^T measures similarity between tokens\nHigher values = more attention",
            font_size=24,
            line_spacing=1.2
        )
        explanation.next_to(VGroup(q_matrix, k_transpose, result_matrix), DOWN, buff=0.8)
        
        return VGroup(
            q_matrix, k_transpose, mult_symbol, result_matrix, equal_sign,
            q_dim, k_dim, result_dim, explanation
        )
    
    def create_matrix(self, label, rows, cols, color=WHITE):
        """Create a visualization of a matrix."""
        cell_size = 0.3
        matrix = VGroup()
        
        # Create grid of cells
        for i in range(rows):
            for j in range(cols):
                cell = Square(side_length=cell_size)
                cell.set_stroke(color, width=1)
                cell.move_to([j*cell_size, -i*cell_size, 0])
                matrix.add(cell)
        
        # Add matrix label
        matrix_label = Text(label, font_size=24, color=color)
        matrix_label.next_to(matrix, UP, buff=0.2)
        
        return VGroup(matrix, matrix_label)
    
    def create_softmax_visualization(self):
        """Create a visualization of the softmax transformation."""
        # Input values
        input_values = [2.5, 1.2, 0.8, 3.1]
        
        # Calculate softmax
        exp_values = [np.exp(x) for x in input_values]
        sum_exp = sum(exp_values)
        softmax_values = [e / sum_exp for e in exp_values]
        
        # Input array
        input_array = self.create_array(input_values, "Input")
        
        # Exponential array
        exp_array = self.create_array(exp_values, "exp(Input)")
        exp_array.next_to(input_array, RIGHT, buff=1)
        
        # Softmax array
        softmax_array = self.create_array(softmax_values, "Softmax")
        softmax_array.next_to(exp_array, RIGHT, buff=1)
        
        # Arrows
        arrows = VGroup(
            Arrow(input_array.get_right(), exp_array.get_left(), buff=0.2),
            Arrow(exp_array.get_right(), softmax_array.get_left(), buff=0.2)
        )
        
        # Arrow labels
        exp_label = Text("Take exp", font_size=20)
        exp_label.next_to(arrows[0], UP, buff=0.1)
        
        normalize_label = Text("Normalize", font_size=20)
        normalize_label.next_to(arrows[1], UP, buff=0.1)
        
        # Sum is 1 annotation
        sum_label = Text("Sum = 1.0", font_size=20, color=YELLOW)
        sum_label.next_to(softmax_array, DOWN, buff=0.3)
        
        return VGroup(
            input_array, exp_array, softmax_array, arrows, exp_label, normalize_label, sum_label
        )
    
    def create_array(self, values, label):
        """Create a visualization of a 1D array with values."""
        cell_size = 0.5
        array = VGroup()
        
        # Create cells with values
        for i, val in enumerate(values):
            cell = Square(side_length=cell_size)
            cell.set_stroke(WHITE, width=1)
            cell.move_to([0, -i*cell_size, 0])
            
            # Add value text
            if abs(val) < 0.01:
                val_text = "≈0"
            elif abs(val) > 999:
                val_text = f"{val:.1e}"
            else:
                val_text = f"{val:.2f}"
                
            value_text = Text(val_text, font_size=16)
            value_text.move_to(cell.get_center())
            
            array.add(VGroup(cell, value_text))
        
        # Add array label
        array_label = Text(label, font_size=24)
        array_label.next_to(array, UP, buff=0.2)
        
        return VGroup(array, array_label)

class MixtureOfExperts(Scene):
    """Explaining the Mixture-of-Experts feed-forward network."""
    
    def construct(self):
        # Title
        title = Text("Mixture-of-Experts Feed-Forward Network", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Explain the concept
        concept = Text(
            "Instead of a single feed-forward network,\nuse multiple specialized 'expert' networks",
            font_size=36,
            line_spacing=1.2
        )
        concept.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(concept, run_time=2))
        self.wait()
        
        # Create a standard FFN for comparison
        standard_title = Text("Standard Feed-Forward Network:", font_size=32)
        standard_title.next_to(concept, DOWN, buff=0.8).to_edge(LEFT)
        
        self.play(Write(standard_title))
        
        standard_ffn = self.create_standard_ffn()
        standard_ffn.next_to(standard_title, DOWN, buff=0.4)
        
        self.play(Create(standard_ffn, run_time=2))
        self.wait()
        
        standard_formula = MathTex(
            r"\text{FFN}(x) = W_2(\text{GELU}(W_1 x + b_1)) + b_2",
            font_size=32
        )
        standard_formula.next_to(standard_ffn, DOWN, buff=0.5)
        
        self.play(Write(standard_formula))
        self.wait(2)
        
        # Transition to MoE
        self.play(
            FadeOut(standard_ffn),
            FadeOut(standard_formula)
        )
        
        moe_title = Text("Mixture-of-Experts Approach:", font_size=32)
        moe_title.next_to(standard_title, RIGHT, buff=3)
        
        self.play(Write(moe_title))
        
        moe_diagram = self.create_moe_diagram()
        moe_diagram.next_to(moe_title, DOWN, buff=0.4)
        
        self.play(Create(moe_diagram, run_time=3))
        self.wait()
        
        # Show MoE formula
        moe_formula = MathTex(
            r"\text{MoE-FFN}(x) = \sum_{e=1}^{4} g_e(x) \cdot \text{FFN}_e(x)",
            font_size=32
        )
        moe_formula.next_to(moe_diagram, DOWN, buff=0.5)
        
        self.play(Write(moe_formula))
        self.wait()
        
        # Show gating formula
        gating_formula = MathTex(
            r"g(x) = \text{softmax}(W_g x)",
            font_size=32
        )
        gating_formula.next_to(moe_formula, DOWN, buff=0.5)
        
        self.play(Write(gating_formula))
        self.wait(2)
        
        # Explain biological significance
        self.play(
            FadeOut(standard_title),
            FadeOut(moe_title),
            FadeOut(moe_diagram),
            FadeOut(concept)
        )
        
        bio_title = Text("Biological Significance of Mixture-of-Experts", font_size=40)
        bio_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(bio_title))
        
        # Create visuals for specialized experts
        experts = self.create_expert_specialization()
        experts.next_to(bio_title, DOWN, buff=0.8)
        
        self.play(Create(experts, run_time=3))
        self.wait()
        
        # Show example of gating
        example_title = Text("Example: How experts process different patients", font_size=32)
        example_title.next_to(experts, DOWN, buff=0.8)
        
        self.play(Write(example_title))
        
        patient_example = self.create_patient_example()
        patient_example.next_to(example_title, DOWN, buff=0.5)
        
        self.play(Create(patient_example, run_time=2))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(bio_title),
            FadeOut(experts),
            FadeOut(example_title),
            FadeOut(patient_example),
            FadeOut(moe_formula),
            FadeOut(gating_formula),
            run_time=1.5
        )
    
    def create_standard_ffn(self):
        """Create a visualization of a standard feed-forward network."""
        # Input node
        input_circle = Circle(radius=0.3, color=WHITE)
        input_text = Text("x", font_size=24)
        input_text.move_to(input_circle.get_center())
        input_node = VGroup(input_circle, input_text)
        
        # Hidden layer nodes
        hidden_nodes = VGroup()
        for i in range(4):
            circle = Circle(radius=0.3, color=BLUE)
            circle.move_to([0, -i*0.8 - 1.5, 0])
            text = Text(f"h{i+1}", font_size=20)
            text.move_to(circle.get_center())
            hidden_nodes.add(VGroup(circle, text))
        
        # Output node
        output_circle = Circle(radius=0.3, color=GREEN)
        output_circle.move_to([0, -5, 0])
        output_text = Text("y", font_size=24)
        output_text.move_to(output_circle.get_center())
        output_node = VGroup(output_circle, output_text)
        
        # Connect with arrows
        arrows1 = VGroup()
        for node in hidden_nodes:
            arrow = Arrow(input_node.get_bottom(), node.get_top(), buff=0.1)
            arrows1.add(arrow)
        
        arrows2 = VGroup()
        for node in hidden_nodes:
            arrow = Arrow(node.get_bottom(), output_node.get_top(), buff=0.1)
            arrows2.add(arrow)
        
        # Layer labels
        input_label = Text("Input", font_size=20)
        input_label.next_to(input_node, LEFT, buff=0.5)
        
        hidden_label = Text("Hidden Layer\n(GELU Activation)", font_size=20, line_spacing=1)
        hidden_label.next_to(hidden_nodes, LEFT, buff=0.5)
        
        output_label = Text("Output", font_size=20)
        output_label.next_to(output_node, LEFT, buff=0.5)
        
        # Weight labels
        w1_label = Text("W1", font_size=18, color=BLUE)
        w1_label.next_to(arrows1[1], RIGHT, buff=0.1)
        
        w2_label = Text("W2", font_size=18, color=GREEN)
        w2_label.next_to(arrows2[1], RIGHT, buff=0.1)
        
        return VGroup(
            input_node, hidden_nodes, output_node, arrows1, arrows2,
            input_label, hidden_label, output_label, w1_label, w2_label
        )
    
    def create_moe_diagram(self):
        """Create a visualization of a mixture-of-experts feed-forward network."""
        # Input
        input_circle = Circle(radius=0.3, color=WHITE)
        input_text = Text("x", font_size=24)
        input_text.move_to(input_circle.get_center())
        input_node = VGroup(input_circle, input_text)
        
        # Gating network
        gate_rect = Rectangle(width=1.5, height=0.8, color=YELLOW)
        gate_text = Text("Gating", font_size=20)
        gate_text.move_to(gate_rect.get_center())
        gate_network = VGroup(gate_rect, gate_text)
        gate_network.next_to(input_node, DOWN, buff=1)
        
        # Expert networks
        experts = VGroup()
        expert_colors = [RED, GREEN, BLUE, PURPLE]
        
        for i in range(4):
            expert = self.create_mini_ffn(expert_colors[i])
            experts.add(expert)
        
        # Position experts
        for i, expert in enumerate(experts):
            expert.move_to([i*2 - 3, -3, 0])
        
        # Arrows from input to gate
        input_gate_arrow = Arrow(input_node.get_bottom(), gate_network.get_top(), buff=0.1)
        
        # Gating output (softmax weights)
        weights_text = VGroup()
        weight_arrows = VGroup()
        
        for i in range(4):
            text = MathTex(f"g_{i+1}", font_size=24, color=YELLOW)
            text.next_to(experts[i], UP, buff=0.8)
            weights_text.add(text)
            
            arrow = Arrow(gate_network.get_bottom(), text.get_top(), buff=0.1, color=YELLOW)
            weight_arrows.add(arrow)
        
        # Input to experts
        input_expert_arrows = VGroup()
        
        for expert in experts:
            arrow = Arrow(input_node.get_center(), expert.get_top(), buff=0.1)
            input_expert_arrows.add(arrow)
        
        # Weighted combination
        output_circle = Circle(radius=0.4, color=WHITE)
        output_text = Text("Σ", font_size=30)
        output_text.move_to(output_circle.get_center())
        output_node = VGroup(output_circle, output_text)
        output_node.next_to(experts, DOWN, buff=1.5)
        
        # Arrows from experts to output
        expert_output_arrows = VGroup()
        
        for i, expert in enumerate(experts):
            arrow = Arrow(expert.get_bottom(), output_node.get_top(), buff=0.1, color=expert_colors[i])
            expert_output_arrows.add(arrow)
        
        # Labels
        gate_label = Text("Determines which\nexperts to use", font_size=18, line_spacing=1)
        gate_label.next_to(gate_network, LEFT, buff=0.5)
        
        output_label = Text("Weighted sum\nof outputs", font_size=18, line_spacing=1)
        output_label.next_to(output_node, RIGHT, buff=0.5)
        
        expert_labels = VGroup()
        for i, expert in enumerate(experts):
            label = Text(f"Expert {i+1}", font_size=18, color=expert_colors[i])
            label.next_to(expert, DOWN, buff=0.2)
            expert_labels.add(label)
        
        return VGroup(
            input_node, gate_network, experts, output_node,
            input_gate_arrow, weight_arrows, input_expert_arrows, expert_output_arrows,
            weights_text, gate_label, output_label, expert_labels
        )
    
    def create_mini_ffn(self, color=BLUE):
        """Create a small feed-forward network for visualization."""
        # Two-layer network
        nodes = VGroup()
        
        # Input node
        input_node = Circle(radius=0.15, color=WHITE)
        
        # Hidden nodes
        for i in range(3):
            node = Circle(radius=0.15, color=color)
            node.move_to([i*0.4 - 0.4, -0.4, 0])
            nodes.add(node)
        
        # Output node
        output_node = Circle(radius=0.15, color=color)
        output_node.move_to([0, -0.8, 0])
        
        # Connect with lines
        lines = VGroup()
        
        for node in nodes:
            line = Line(input_node.get_center(), node.get_center(), color=color, stroke_width=1)
            lines.add(line)
        
        for node in nodes:
            line = Line(node.get_center(), output_node.get_center(), color=color, stroke_width=1)
            lines.add(line)
        
        mini_ffn = VGroup(input_node, nodes, output_node, lines)
        
        return mini_ffn
    
    def create_expert_specialization(self):
        """Create a visualization of expert specializations."""
        # Create four specialized "experts"
        specializations = VGroup()
        
        # Expert 1: Immune genes
        immune_rect = Rectangle(width=3, height=1.5, color=RED)
        immune_title = Text("Expert 1: Immune Genes", font_size=24, color=RED)
        immune_title.next_to(immune_rect, UP, buff=0.2)
        immune_content = Text("HLA-DRB1, IFNG, TNF", font_size=20)
        immune_content.move_to(immune_rect.get_center())
        immune_group = VGroup(immune_rect, immune_title, immune_content)
        
        # Expert 2: Stress response
        stress_rect = Rectangle(width=3, height=1.5, color=GREEN)
        stress_title = Text("Expert 2: Stress Response", font_size=24, color=GREEN)
        stress_title.next_to(stress_rect, UP, buff=0.2)
        stress_content = Text("NR3C1, CRH, FKBP5", font_size=20)
        stress_content.move_to(stress_rect.get_center())
        stress_group = VGroup(stress_rect, stress_title, stress_content)
        stress_group.next_to(immune_group, RIGHT, buff=1)
        
        # Expert 3: Metabolic pathways
        metabolic_rect = Rectangle(width=3, height=1.5, color=BLUE)
        metabolic_title = Text("Expert 3: Metabolism", font_size=24, color=BLUE)
        metabolic_title.next_to(metabolic_rect, UP, buff=0.2)
        metabolic_content = Text("PDK2, AMPK, MTOR", font_size=20)
        metabolic_content.move_to(metabolic_rect.get_center())
        metabolic_group = VGroup(metabolic_rect, metabolic_title, metabolic_content)
        metabolic_group.next_to(immune_group, DOWN, buff=1)
        
        # Expert 4: Viral response
        viral_rect = Rectangle(width=3, height=1.5, color=PURPLE)
        viral_title = Text("Expert 4: Viral Response", font_size=24, color=PURPLE)
        viral_title.next_to(viral_rect, UP, buff=0.2)
        viral_content = Text("IFITM3, ISG15, OAS1", font_size=20)
        viral_content.move_to(viral_rect.get_center())
        viral_group = VGroup(viral_rect, viral_title, viral_content)
        viral_group.next_to(stress_group, DOWN, buff=1)
        
        specializations.add(immune_group, stress_group, metabolic_group, viral_group)
        
        return specializations
    
    def create_patient_example(self):
        """Create an example of different gating for different patients."""
        # Create a table showing expert weights for different patients
        table = VGroup()
        
        # Header row
        header = VGroup()
        header_texts = ["Patient Type", "Expert 1\n(Immune)", "Expert 2\n(Stress)", "Expert 3\n(Metabolism)", "Expert 4\n(Viral)"]
        
        for i, text in enumerate(header_texts):
            cell = Rectangle(width=2, height=0.8)
            cell.set_stroke(WHITE, width=1)
            if i > 0:
                cell.next_to(header[i-1], RIGHT, buff=0)
            
            cell_text = Text(text, font_size=16, line_spacing=0.8)
            cell_text.move_to(cell.get_center())
            
            header.add(VGroup(cell, cell_text))
        
        table.add(header)
        
        # Data rows
        patient_types = ["ME/CFS", "Long COVID", "Control"]
        
        # Expert weights for different patients (normalized to sum to 1)
        weights = [
            [0.45, 0.30, 0.20, 0.05],  # ME/CFS - high immune & stress
            [0.30, 0.15, 0.15, 0.40],  # Long COVID - high immune & viral
            [0.20, 0.25, 0.40, 0.15]   # Control - more balanced, higher metabolism
        ]
        
        for i, patient in enumerate(patient_types):
            row = VGroup()
            
            # Patient cell
            patient_cell = Rectangle(width=2, height=0.8)
            patient_cell.set_stroke(WHITE, width=1)
            patient_cell.next_to(header[0], DOWN, buff=0) if i == 0 else patient_cell.next_to(table[-1][0], DOWN, buff=0)
            
            patient_text = Text(patient, font_size=16)
            patient_text.move_to(patient_cell.get_center())
            
            row.add(VGroup(patient_cell, patient_text))
            
            # Weight cells
            for j, weight in enumerate(weights[i]):
                cell = Rectangle(width=2, height=0.8)
                cell.set_stroke(WHITE, width=1)
                cell.next_to(row[j], RIGHT, buff=0)
                
                # Color-coded weight
                color = self.get_weight_color(weight)
                weight_text = Text(f"{weight:.2f}", font_size=16, color=color)
                weight_text.move_to(cell.get_center())
                
                row.add(VGroup(cell, weight_text))
            
            table.add(row)
        
        # Highlight the highest weight in each row
        highlights = VGroup()
        
        for i, row in enumerate(table[1:]):
            max_idx = weights[i].index(max(weights[i]))
            cell = row[max_idx + 1][0]  # +1 to skip patient name cell
            highlight = SurroundingRectangle(cell, color=YELLOW, buff=0.05)
            highlights.add(highlight)
        
        # Add explanation
        explanation = Text(
            "The gating mechanism routes each sample to the most relevant experts",
            font_size=24
        )
        explanation.next_to(table, DOWN, buff=0.8)
        
        return VGroup(table, highlights, explanation)
    
    def get_weight_color(self, weight):
        """Color code for weights (higher = more saturated)."""
        if weight < 0.2:
            return GRAY
        elif weight < 0.3:
            return BLUE_C
        elif weight < 0.4:
            return GREEN_C
        else:
            return RED_C

class AdaptiveComputationTime(Scene):
    """Explaining the Adaptive Computation Time mechanism."""
    
    def construct(self):
        # Title
        title = Text("Adaptive Computation Time (ACT)", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Explain the concept
        concept = Text(
            "Key idea: Allow the model to decide how many\nprocessing steps to use for each sample",
            font_size=36,
            line_spacing=1.2
        )
        concept.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(concept, run_time=2))
        self.wait()
        
        # Visualize the ACT mechanism
        act_diagram = self.create_act_diagram()
        act_diagram.next_to(concept, DOWN, buff=0.8)
        
        self.play(Create(act_diagram, run_time=3))
        self.wait()
        
        # Show the halting probability calculation
        halt_title = Text("Halting Probability Calculation:", font_size=32)
        halt_title.next_to(act_diagram, DOWN, buff=0.8)
        
        self.play(Write(halt_title))
        
        halt_formula = MathTex(r"p = \sigma(w^T x_{\text{mean}})", font_size=36)
        halt_formula.next_to(halt_title, DOWN, buff=0.4)
        
        self.play(Write(halt_formula))
        self.wait()
        
        # Show the halting mechanism
        halting_mechanism = self.create_halting_mechanism()
        halting_mechanism.next_to(halt_formula, DOWN, buff=0.8)
        
        self.play(Create(halting_mechanism, run_time=2))
        self.wait(2)
        
        # Explain biological relevance
        self.play(
            FadeOut(act_diagram),
            FadeOut(halt_title),
            FadeOut(halt_formula),
            FadeOut(halting_mechanism)
        )
        
        relevance_title = Text("Clinical Relevance:", font_size=36)
        relevance_title.next_to(concept, DOWN, buff=0.8)
        
        self.play(Write(relevance_title))
        
        relevance = self.create_relevance_viz()
        relevance.next_to(relevance_title, DOWN, buff=0.4)
        
        self.play(Create(relevance, run_time=2))
        self.wait()
        
        # Show distribution of iterations
        distribution_title = Text("Distribution of Iterations in the Model:", font_size=32)
        distribution_title.next_to(relevance, DOWN, buff=0.8)
        
        self.play(Write(distribution_title))
        
        distribution = self.create_iteration_distribution()
        distribution.next_to(distribution_title, DOWN, buff=0.4)
        
        self.play(Create(distribution))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(concept),
            FadeOut(relevance_title),
            FadeOut(relevance),
            FadeOut(distribution_title),
            FadeOut(distribution),
            run_time=1.5
        )
    
    def create_act_diagram(self):
        """Create a visualization of the Adaptive Computation Time mechanism."""
        # Input
        input_rect = Rectangle(width=3, height=0.8, color=WHITE)
        input_text = Text("Sample Features", font_size=24)
        input_text.move_to(input_rect.get_center())
        input_group = VGroup(input_rect, input_text)
        
        # Transformer encoder
        encoder_rect = Rectangle(width=3, height=1.5, color=TRANSFORMER_COLOR)
        encoder_text = Text("Transformer\nEncoder", font_size=24, line_spacing=1)
        encoder_text.move_to(encoder_rect.get_center())
        encoder_group = VGroup(encoder_rect, encoder_text)
        encoder_group.next_to(input_group, DOWN, buff=0.8)
        
        # Halting unit
        halt_rect = Rectangle(width=2, height=0.8, color=RED)
        halt_text = Text("Halting Unit", font_size=20)
        halt_text.move_to(halt_rect.get_center())
        halt_group = VGroup(halt_rect, halt_text)
        halt_group.next_to(encoder_group, RIGHT, buff=1.5)
        
        # Output
        output_rect = Rectangle(width=3, height=0.8, color=GREEN)
        output_text = Text("Output", font_size=24)
        output_text.move_to(output_rect.get_center())
        output_group = VGroup(output_rect, output_text)
        output_group.next_to(encoder_group, DOWN, buff=0.8)
        
        # Arrows
        input_encoder = Arrow(input_group.get_bottom(), encoder_group.get_top(), buff=0.1)
        encoder_output = Arrow(encoder_group.get_bottom(), output_group.get_top(), buff=0.1)
        encoder_halt = Arrow(encoder_group.get_right(), halt_group.get_left(), buff=0.1)
        
        # Loop back arrow
        loop_arrow = CurvedArrow(
            halt_group.get_bottom(),
            encoder_group.get_top() + RIGHT * 0.5,
            angle=-np.pi/2
        )
        
        # Decision diamond
        decision = Polygon(
            UP * 0.4, RIGHT * 0.4, DOWN * 0.4, LEFT * 0.4,
            color=YELLOW
        )
        decision.next_to(halt_group, DOWN, buff=0.5)
        decision_text = Text("p>0.99?", font_size=18)
        decision_text.move_to(decision.get_center())
        decision_group = VGroup(decision, decision_text)
        
        halt_to_decision = Arrow(halt_group.get_bottom(), decision.get_top(), buff=0.1)
        
        yes_arrow = Arrow(decision.get_right(), decision.get_right() + RIGHT * 0.8, buff=0.1)
        yes_text = Text("Yes", font_size=18, color=GREEN)
        yes_text.next_to(yes_arrow, UP, buff=0.1)
        
        no_arrow = Arrow(decision.get_bottom(), loop_arrow.get_start(), buff=0.1)
        no_text = Text("No", font_size=18, color=RED)
        no_text.next_to(no_arrow, RIGHT, buff=0.1)
        
        # Add labels
        max_iter = Text("Max iterations = 3", font_size=20, color=RED)
        max_iter.next_to(loop_arrow, LEFT, buff=0.5)
        
        return VGroup(
            input_group, encoder_group, halt_group, output_group,
            input_encoder, encoder_output, encoder_halt,
            loop_arrow, decision_group, halt_to_decision,
            yes_arrow, yes_text, no_arrow, no_text, max_iter
        )
    
    def create_halting_mechanism(self):
        """Create a visualization of the halting mechanism details."""
        # Table showing halting probability accumulation
        table = VGroup()
        
        # Header
        header_texts = ["Iteration", "p", "H (accumulated)", "Action"]
        cells = VGroup()
        
        for i, text in enumerate(header_texts):
            cell = Rectangle(width=2, height=0.6)
            cell.set_stroke(WHITE, width=1)
            if i > 0:
                cell.next_to(cells[-1], RIGHT, buff=0)
            else:
                cell.to_edge(LEFT, buff=1)
            
            cell_text = Text(text, font_size=20)
            cell_text.move_to(cell.get_center())
            
            cells.add(VGroup(cell, cell_text))
        
        table.add(cells)
        
        # Example rows
        example_data = [
            ["1", "0.3", "0.3", "Continue"],
            ["2", "0.6", "0.9", "Continue"],
            ["3", "0.2", "1.0", "Halt (H ≥ 1)"]
        ]
        
        for data in example_data:
            row = VGroup()
            
            for i, value in enumerate(data):
                cell = Rectangle(width=2, height=0.6)
                cell.set_stroke(WHITE, width=1)
                
                if i == 0:  # First cell in row
                    prev_row = table[-1] if len(table) > 1 else table[0]
                    cell.next_to(prev_row[0], DOWN, buff=0)
                else:
                    cell.next_to(row[-1], RIGHT, buff=0)
                
                color = WHITE
                if i == 3:  # Action column
                    color = GREEN if "Halt" in value else BLUE
                
                cell_text = Text(value, font_size=18, color=color)
                cell_text.move_to(cell.get_center())
                
                row.add(VGroup(cell, cell_text))
            
            table.add(row)
        
        # Add explanation
        explanation = Text(
            "Halting probability accumulates until reaching 1.0\nOr maximum iterations is reached",
            font_size=24,
            line_spacing=1
        )
        explanation.next_to(table, DOWN, buff=0.5)
        
        # Add loss penalty term
        penalty = MathTex(
            r"R = \sum_{\text{sample } j} \sum_{t=1}^{T_j} \big( p_j^{(t)} \prod_{s< t}(1 - p_j^{(s)}) \big) \cdot t",
            font_size=28
        )
        penalty.next_to(explanation, DOWN, buff=0.5)
        
        penalty_explanation = Text(
            "Loss term penalizes excessive computation",
            font_size=20,
            color=RED
        )
        penalty_explanation.next_to(penalty, DOWN, buff=0.3)
        
        return VGroup(table, explanation, penalty, penalty_explanation)
    
    def create_relevance_viz(self):
        """Create a visualization of the clinical relevance of ACT."""
        # Create two patient examples
        example = VGroup()
        
        # Patient 1: Clear case
        clear_rect = Rectangle(width=4, height=2, color=BLUE)
        clear_title = Text("Clear Case Patient", font_size=28, color=BLUE)
        clear_title.next_to(clear_rect, UP, buff=0.2)
        
        clear_content = VGroup(
            Text("• Strong methylation signature", font_size=20),
            Text("• Clear disease patterns", font_size=20),
            Text("• Halts after 1 iteration (p=0.95)", font_size=20, color=GREEN)
        )
        clear_content.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        clear_content.move_to(clear_rect.get_center())
        
        clear_group = VGroup(clear_rect, clear_title, clear_content)
        
        # Patient 2: Ambiguous case
        ambig_rect = Rectangle(width=4, height=2, color=RED)
        ambig_title = Text("Ambiguous Case Patient", font_size=28, color=RED)
        ambig_title.next_to(ambig_rect, UP, buff=0.2)
        
        ambig_content = VGroup(
            Text("• Mixed methylation patterns", font_size=20),
            Text("• Borderline between conditions", font_size=20),
            Text("• Needs 3 iterations (p=0.3, 0.4, 0.3)", font_size=20, color=YELLOW)
        )
        ambig_content.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        ambig_content.move_to(ambig_rect.get_center())
        
        ambig_group = VGroup(ambig_rect, ambig_title, ambig_content)
        ambig_group.next_to(clear_group, RIGHT, buff=1)
        
        example.add(clear_group, ambig_group)
        
        # Add explanation
        explanation = Text(
            "More computation is allocated to difficult or ambiguous cases,\nmaking the model efficient and accurate.",
            font_size=24,
            line_spacing=1
        )
        explanation.next_to(VGroup(clear_group, ambig_group), DOWN, buff=0.5)
        
        return VGroup(example, explanation)
    
    def create_iteration_distribution(self):
        """Create a bar chart showing distribution of iterations."""
        # Create axes
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 80, 20],
            x_length=6,
            y_length=3,
            axis_config={"color": WHITE}
        )
        
        # Labels
        x_label = Text("Number of Iterations", font_size=20)
        x_label.next_to(axes, DOWN, buff=0.3)
        
        y_label = Text("% of Samples", font_size=20)
        y_label.next_to(axes, LEFT, buff=0.3).rotate(PI/2)
        
        # Data
        data = [0, 70, 25, 5]  # 0%, 70%, 25%, 5% for 0, 1, 2, 3 iterations
        
        # Create bars
        bars = VGroup()
        bar_colors = [BLUE_E, BLUE, YELLOW, RED]
        
        for i, value in enumerate(data):
            if i > 0:  # Skip the 0 iteration case
                bar = Rectangle(
                    width=0.5,
                    height=value * 3 / 100,  # Scale to match y-axis
                    fill_opacity=0.8,
                    fill_color=bar_colors[i],
                    stroke_color=WHITE
                )
                bar.move_to(axes.c2p(i, value/2), aligned_edge=DOWN)
                bars.add(bar)
                
                # Add value label
                value_label = Text(f"{value}%", font_size=16)
                value_label.next_to(bar, UP, buff=0.1)
                bars.add(value_label)
        
        # Add legend
        legend = VGroup()
        
        for i in range(1, 4):
            color = bar_colors[i]
            square = Square(side_length=0.2, fill_color=color, fill_opacity=0.8)
            label = Text(f"{i} iteration{'s' if i > 1 else ''}", font_size=16)
            label.next_to(square, RIGHT, buff=0.2)
            group = VGroup(square, label)
            
            if i > 1:
                group.next_to(legend[-1], RIGHT, buff=0.5)
            
            legend.add(group)
        
        legend.arrange(RIGHT, buff=0.5)
        legend.next_to(axes, UP, buff=0.5)
        
        return VGroup(axes, x_label, y_label, bars, legend)

class MaskedPretraining(Scene):
    """Explaining the masked pretraining technique."""
    
    def construct(self):
        # Title
        title = Text("Self-Supervised Masked Pretraining", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Explain the concept
        concept = Text(
            "Train the model to reconstruct masked methylation values\nbefore fine-tuning for classification",
            font_size=36,
            line_spacing=1.2
        )
        concept.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(concept, run_time=2))
        self.wait()
        
        # Show the masking process
        masking_title = Text("Methylation Value Masking Process:", font_size=32)
        masking_title.next_to(concept, DOWN, buff=0.8)
        
        self.play(Write(masking_title))
        
        masking_viz = self.create_masking_visualization()
        masking_viz.next_to(masking_title, DOWN, buff=0.4)
        
        self.play(Create(masking_viz, run_time=2))
        self.wait()
        
        # Show the loss function
        loss_title = Text("Pretraining Loss Function:", font_size=32)
        loss_title.next_to(masking_viz, DOWN, buff=0.8)
        
        self.play(Write(loss_title))
        
        loss_formula = MathTex(
            r"\mathcal{L}_{\text{MSE}} = \frac{1}{\sum_{i,j} M_{ij}} \sum_{i,j} M_{ij} (\hat{x}_{ij} - x_{ij})^2",
            font_size=32
        )
        loss_formula.next_to(loss_title, DOWN, buff=0.4)
        
        self.play(Write(loss_formula))
        self.wait()
        
        # Show pretraining curve
        curve_title = Text("Pretraining Convergence:", font_size=32)
        curve_title.next_to(loss_formula, DOWN, buff=0.8)
        
        self.play(Write(curve_title))
        
        pretrain_curve = self.create_pretrain_curve()
        pretrain_curve.next_to(curve_title, DOWN, buff=0.4)
        
        self.play(Create(pretrain_curve, run_time=2))
        self.wait(2)
        
        # Explain benefits
        self.play(
            FadeOut(masking_viz),
            FadeOut(loss_title),
            FadeOut(loss_formula),
            FadeOut(curve_title),
            FadeOut(pretrain_curve)
        )
        
        benefits_title = Text("Benefits of Masked Pretraining:", font_size=36)
        benefits_title.next_to(masking_title, DOWN, buff=0.8)
        
        self.play(Write(benefits_title))
        
        benefits = self.create_benefits_viz()
        benefits.next_to(benefits_title, DOWN, buff=0.4)
        
        self.play(Create(benefits, run_time=2))
        self.wait()
        
        # Results comparing with and without pretraining
        results_title = Text("Impact on Classification Performance:", font_size=32)
        results_title.next_to(benefits, DOWN, buff=0.8)
        
        self.play(Write(results_title))
        
        results = self.create_results_comparison()
        results.next_to(results_title, DOWN, buff=0.4)
        
        self.play(Create(results))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(concept),
            FadeOut(masking_title),
            FadeOut(benefits_title),
            FadeOut(benefits),
            FadeOut(results_title),
            FadeOut(results),
            run_time=1.5
        )
    
    def create_masking_visualization(self):
        """Create a visualization of the masking process."""
        # Original methylation values
        original_values = np.random.rand(10)
        original_vec = self.create_methylation_vector(original_values, "Original")
        
        # Masked values (replace 15% with special token)
        masked_values = original_values.copy()
        mask_indices = [1, 5, 8]  # 30% just for visualization clarity
        for i in mask_indices:
            masked_values[i] = 0
        
        # Arrow
        arrow = Arrow(UP * 0.5, DOWN * 0.5, buff=0.3)
        arrow.next_to(original_vec, DOWN, buff=0.5)
        
        arrow_label = Text("Mask 15% of values", font_size=24)
        arrow_label.next_to(arrow, RIGHT, buff=0.3)
        
        # Masked vector
        masked_vec = self.create_methylation_vector(masked_values, "Masked", mask_indices)
        masked_vec.next_to(arrow, DOWN, buff=0.5)
        
        # Transformer prediction
        prediction_arrow = Arrow(UP * 0.5, DOWN * 0.5, buff=0.3)
        prediction_arrow.next_to(masked_vec, DOWN, buff=0.5)
        
        arrow_label2 = Text("Transformer predicts", font_size=24)
        arrow_label2.next_to(prediction_arrow, RIGHT, buff=0.3)
        
        # Predicted vector
        predicted_values = original_values.copy()
        # Add some error to predictions
        for i in mask_indices:
            predicted_values[i] = original_values[i] + (np.random.rand() - 0.5) * 0.2
            predicted_values[i] = max(0, min(1, predicted_values[i]))  # Clamp to [0,1]
        
        predicted_vec = self.create_methylation_vector(predicted_values, "Predicted", mask_indices, prediction=True)
        predicted_vec.next_to(prediction_arrow, DOWN, buff=0.5)
        
        return VGroup(
            original_vec, arrow, arrow_label, masked_vec,
            prediction_arrow, arrow_label2, predicted_vec
        )
    
    def create_methylation_vector(self, values, label_text, mask_indices=None, prediction=False):
        """Create a visualization of a methylation vector."""
        n_values = len(values)
        square_size = 0.4
        gap = 0.05
        
        vector = VGroup()
        
        # Create squares with values
        for i, val in enumerate(values):
            color = self.get_methylation_color(val)
            square = Square(side_length=square_size)
            
            if mask_indices is not None and i in mask_indices:
                if prediction:
                    square.set_fill(color, opacity=0.8)
                    square.set_stroke(YELLOW, width=3)  # Highlight predictions
                else:
                    square.set_fill(GRAY, opacity=0.5)
                    square.set_stroke(WHITE, width=1)
            else:
                square.set_fill(color, opacity=0.8)
                square.set_stroke(WHITE, width=1)
            
            square.move_to([i * (square_size + gap), 0, 0])
            
            # Add value text
            if mask_indices is not None and i in mask_indices and not prediction:
                value_text = Text("?", font_size=18)
            else:
                value_text = Text(f"{val:.2f}", font_size=12)
            
            value_text.move_to(square.get_center())
            
            vector.add(VGroup(square, value_text))
        
        # Add vector label
        vector_label = Text(label_text, font_size=24)
        vector_label.next_to(vector, LEFT, buff=0.5)
        
        return VGroup(vector, vector_label)
    
    def get_methylation_color(self, value):
        """Map methylation value to color (blue to red gradient)."""
        r = value
        g = 0.2
        b = 1 - value
        return rgb_to_color([r, g, b])
    
    def create_pretrain_curve(self):
        """Create a visualization of the pretraining convergence curve."""
        # Create axes
        axes = Axes(
            x_range=[0, 30, 5],
            y_range=[0, 1.2, 0.2],
            x_length=6,
            y_length=3,
            axis_config={"color": WHITE}
        )
        
        # Labels
        x_label = Text("Epochs", font_size=24)
        x_label.next_to(axes, DOWN, buff=0.3)
        
        y_label = Text("MSE Loss", font_size=24)
        y_label.next_to(axes, LEFT, buff=0.3).rotate(PI/2)
        
        # Create curve
        def loss_function(x):
            return 1.1 * np.exp(-0.15 * x) + 0.08
        
        x_vals = np.linspace(0, 30, 100)
        y_vals = [loss_function(x) for x in x_vals]
        
        curve = VGroup()
        
        for i in range(len(x_vals) - 1):
            line = Line(
                axes.c2p(x_vals[i], y_vals[i]),
                axes.c2p(x_vals[i+1], y_vals[i+1]),
                color=BLUE,
                stroke_width=3
            )
            curve.add(line)
        
        # Add point showing convergence
        converge_point = Dot(axes.c2p(30, y_vals[-1]), color=YELLOW, radius=0.08)
        
        converge_label = Text("Final Loss: 0.00009", font_size=20, color=YELLOW)
        converge_label.next_to(converge_point, UP, buff=0.2)
        
        # Add title
        title = Text("Reconstruction MSE during Pretraining", font_size=28)
        title.to_edge(UP, buff=0.2)
        
        return VGroup(axes, x_label, y_label, curve, converge_point, converge_label)
    
    def create_benefits_viz(self):
        """Create a visualization of the benefits of masked pretraining."""
        # Create a list of benefits
        benefits = VGroup(
            Text("1. Learns biologically meaningful patterns without labels", font_size=28),
            Text("2. Initializes weights to better starting point", font_size=28),
            Text("3. Learns co-methylation patterns between CpG sites", font_size=28),
            Text("4. Can leverage unlabeled methylation data", font_size=28)
        )
        
        benefits.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        # Highlight key words
        highlights = VGroup()
        
        keywords = [
            ["biologically", "patterns"],
            ["better", "starting point"],
            ["co-methylation", "patterns"],
            ["unlabeled", "data"]
        ]
        
        for i, txt in enumerate(benefits):
            for keyword in keywords[i]:
                # Find the keyword in the text
                txt_str = txt.text
                start_idx = txt_str.find(keyword)
                if start_idx != -1:
                    # Calculate the position of the keyword in the text
                    char_indices = [j for j, char in enumerate(txt_str) if char == keyword[0]]
                    closest_idx = min(char_indices, key=lambda x: abs(x - start_idx))
                    
                    # Create a highlight rectangle
                    highlight = Rectangle(
                        width=len(keyword) * 0.18,  # Approximate width per character
                        height=0.3,
                        color=YELLOW,
                        fill_opacity=0.3
                    )
                    
                    # Position the highlight over the keyword
                    highlight.move_to(txt.submobjects[closest_idx:closest_idx+len(keyword)])
                    # Position the highlight over the keyword
                    highlight.move_to(txt[closest_idx:closest_idx+len(keyword)])
                    highlights.add(highlight)
        
        return VGroup(benefits, highlights)
    
    def create_results_comparison(self):
        """Create a bar chart comparing results with and without pretraining."""
        # Create axes
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 100, 20],
            x_length=8,
            y_length=4,
            axis_config={"color": WHITE}
        )
        
        # Labels
        x_label = Text("Model Variant", font_size=24)
        x_label.next_to(axes, DOWN, buff=0.3)
        
        y_label = Text("Accuracy (%)", font_size=24)
        y_label.next_to(axes, LEFT, buff=0.3).rotate(PI/2)
        
        # Data
        data = [89.0, 92.0, 97.06]  # No MoE/ACT, No Pretraining, Full Model
        bar_labels = ["Standard\nTransformer", "Without\nPretraining", "With\nPretraining"]
        
        # Create bars
        bars = VGroup()
        bar_colors = [BLUE_E, BLUE, GREEN]
        
        for i, (value, label) in enumerate(zip(data, bar_labels)):
            bar = Rectangle(
                width=0.6,
                height=value * 4 / 100,  # Scale to match y-axis
                fill_opacity=0.8,
                fill_color=bar_colors[i],
                stroke_color=WHITE
            )
            bar.move_to(axes.c2p(i+0.5, value/2), aligned_edge=DOWN)
            bars.add(bar)
            
            # Add value label
            value_label = Text(f"{value}%", font_size=20)
            value_label.next_to(bar, UP, buff=0.1)
            
            # Add bar label
            bar_label = Text(label, font_size=20, line_spacing=0.8)
            bar_label.next_to(bar, DOWN, buff=0.5)
            
            bars.add(value_label, bar_label)
        
        # Highlight the improvement
        improvement = Text("+5.06% from pretraining", font_size=24, color=GREEN)
        improvement.next_to(bars, UP, buff=0.5)
        
        # Add arrow showing improvement
        improvement_arrow = Arrow(
            bars[5].get_top() + UP * 0.3,  # Third bar's label
            bars[8].get_top() + UP * 0.3,  # Last bar's label
            color=GREEN,
            buff=0.1
        )
        improvement_arrow.next_to(improvement, DOWN, buff=0.2)
        
        return VGroup(axes, x_label, y_label, bars, improvement, improvement_arrow)

class ResultsAndConclusion(Scene):
    """Presenting the results and conclusion of the research."""
    
    def construct(self):
        # Title
        title = Text("Results and Biological Insights", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Performance results
        results_title = Text("Classification Performance", font_size=36)
        results_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(results_title))
        
        results_table = self.create_results_table()
        results_table.next_to(results_title, DOWN, buff=0.5)
        
        self.play(Create(results_table, run_time=2))
        self.wait()
        
        # Show the confusion matrix
        matrix_title = Text("Confusion Matrix", font_size=32)
        matrix_title.next_to(results_table, DOWN, buff=0.8)
        
        self.play(Write(matrix_title))
        
        confusion_matrix = self.create_confusion_matrix()
        confusion_matrix.next_to(matrix_title, DOWN, buff=0.4)
        
        self.play(Create(confusion_matrix))
        self.wait(2)
        
        # Transition to biological insights
        self.play(
            FadeOut(results_title),
            FadeOut(results_table),
            FadeOut(matrix_title),
            FadeOut(confusion_matrix)
        )
        
        # Biological insights
        insights_title = Text("Key Epigenetic Biomarkers Discovered", font_size=36)
        insights_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(insights_title))
        
        gene_insights = self.create_gene_insights()
        gene_insights.next_to(insights_title, DOWN, buff=0.5)
        
        self.play(Create(gene_insights, run_time=2))
        self.wait(2)
        
        # Attention visualization
        attention_title = Text("Attention Map Visualization", font_size=32)
        attention_title.next_to(gene_insights, DOWN, buff=0.8)
        
        self.play(Write(attention_title))
        
        attention_map = self.create_attention_map()
        attention_map.next_to(attention_title, DOWN, buff=0.4)
        
        self.play(Create(attention_map, run_time=2))
        self.wait(2)
        
        # Transition to conclusion
        self.play(
            FadeOut(insights_title),
            FadeOut(gene_insights),
            FadeOut(attention_title),
            FadeOut(attention_map)
        )
        
        # Conclusion
        conclusion_title = Text("Conclusions and Impact", font_size=36)
        conclusion_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(conclusion_title))
        
        conclusions = self.create_conclusions()
        conclusions.next_to(conclusion_title, DOWN, buff=0.5)
        
        self.play(Create(conclusions, run_time=2))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(conclusion_title),
            FadeOut(conclusions),
            run_time=1.5
        )
    
    def create_results_table(self):
        """Create a table of classification results."""
        # Headers
        headers = ["Model", "Accuracy", "Macro F1", "Macro AUROC"]
        models = [
            "Logistic Regression",
            "Random Forest",
            "XGBoost",
            "Transformer (ours)"
        ]
        accuracy = ["75.0%", "78.3%", "80.0%", "97.06%"]
        f1 = ["0.73", "0.76", "0.78", "0.95"]
        auroc = ["0.80", "0.82", "0.85", "0.98"]
        
        # Create table
        table = VGroup()
        
        # Header row
        header_row = VGroup()
        for i, text in enumerate(headers):
            cell = Rectangle(width=3, height=0.8)
            cell.set_stroke(WHITE, width=1)
            
            if i > 0:
                cell.next_to(header_row[-1], RIGHT, buff=0)
            
            cell_text = Text(text, font_size=24)
            cell_text.move_to(cell.get_center())
            
            header_row.add(VGroup(cell, cell_text))
        
        table.add(header_row)
        
        # Data rows
        for i in range(len(models)):
            row_data = [models[i], accuracy[i], f1[i], auroc[i]]
            row = VGroup()
            
            for j, text in enumerate(row_data):
                cell = Rectangle(width=3, height=0.8)
                cell.set_stroke(WHITE, width=1)
                
                if j == 0:
                    cell.next_to(table[-1][0], DOWN, buff=0)
                else:
                    cell.next_to(row[-1], RIGHT, buff=0)
                
                # Color the best result
                if i == len(models) - 1 and j > 0:
                    cell_text = Text(text, font_size=24, color=GREEN)
                else:
                    cell_text = Text(text, font_size=24)
                
                cell_text.move_to(cell.get_center())
                
                row.add(VGroup(cell, cell_text))
            
            table.add(row)
        
        return table
    
    def create_confusion_matrix(self):
        """Create a visualization of the confusion matrix."""
        # Matrix data (rows: true class, columns: predicted class)
        # Format: [[MECFS->MECFS, MECFS->LC, MECFS->Control],
        #          [LC->MECFS, LC->LC, LC->Control],
        #          [Control->MECFS, Control->LC, Control->Control]]
        matrix_data = [
            [4, 1, 0],
            [1, 4, 0],
            [0, 0, 10]
        ]
        
        # Class labels
        class_labels = ["ME/CFS", "Long COVID", "Control"]
        
        # Create the matrix
        matrix = VGroup()
        
        # Add row and column headers
        row_headers = VGroup()
        for i, label in enumerate(class_labels):
            header = Text(f"True: {label}", font_size=20)
            header.to_edge(LEFT, buff=1)
            if i > 0:
                header.next_to(row_headers[-1], DOWN, buff=0.8)
            row_headers.add(header)
        
        col_headers = VGroup()
        for i, label in enumerate(class_labels):
            header = Text(f"Predicted: {label}", font_size=20)
            if i > 0:
                header.next_to(col_headers[-1], RIGHT, buff=1.2)
            col_headers.add(header)
        col_headers.arrange(RIGHT, buff=1.2)
        col_headers.next_to(row_headers, UP, buff=0.8)
        
        # Create cells
        cells = VGroup()
        for i in range(len(class_labels)):
            row = VGroup()
            for j in range(len(class_labels)):
                # Set cell color based on correct/incorrect prediction
                if i == j:
                    color = GREEN_E  # Correct prediction
                    opacity = 0.7
                else:
                    color = RED_E  # Incorrect prediction
                    opacity = 0.5
                
                cell = Square(side_length=1)
                cell.set_fill(color, opacity=opacity)
                cell.set_stroke(WHITE, width=1)
                
                # Position cell
                if j > 0:
                    cell.next_to(row[-1], RIGHT, buff=0.2)
                else:
                    cell.next_to(row_headers[i], RIGHT, buff=0.5)
                
                # Add count text
                count_text = Text(str(matrix_data[i][j]), font_size=28)
                count_text.move_to(cell.get_center())
                
                row.add(VGroup(cell, count_text))
            
            cells.add(row)
        
        # Add a note about the confusion matrix
        note = Text("Note: No controls misclassified as patients (high specificity)", 
                   font_size=20, color=GREEN)
        note.next_to(cells, DOWN, buff=0.8)
        
        return VGroup(row_headers, col_headers, cells, note)
    
    def create_gene_insights(self):
        """Create a visualization of the gene/biological insights."""
        # Create a table of top genes and their methylation patterns
        table = VGroup()
        
        # Headers
        headers = ["Gene", "Function", "ME/CFS Pattern", "Long COVID Pattern"]
        header_row = VGroup()
        
        for i, text in enumerate(headers):
            cell = Rectangle(width=3, height=0.8)
            cell.set_stroke(WHITE, width=1)
            
            if i > 0:
                cell.next_to(header_row[-1], RIGHT, buff=0)
            
            cell_text = Text(text, font_size=20)
            cell_text.move_to(cell.get_center())
            
            header_row.add(VGroup(cell, cell_text))
        
        table.add(header_row)
        
        # Gene data
        genes = [
            "HLA-DRB1",
            "NR3C1",
            "IFNG",
            "IFITM3",
            "PDK2"
        ]
        
        functions = [
            "Immune regulation",
            "Stress response",
            "Interferon signaling",
            "Viral defense",
            "Metabolism"
        ]
        
        mecfs_patterns = [
            "Hypomethylated",
            "Hypomethylated (β=0.55)",
            "Differentially methylated",
            "No significant change",
            "Hypermethylated"
        ]
        
        lc_patterns = [
            "Mixed pattern",
            "Variable methylation",
            "Hypomethylated",
            "Hypomethylated (β=0.48)",
            "No significant change"
        ]
        
        # Create rows
        for i in range(len(genes)):
            row_data = [genes[i], functions[i], mecfs_patterns[i], lc_patterns[i]]
            row = VGroup()
            
            for j, text in enumerate(row_data):
                cell = Rectangle(width=3, height=0.8)
                cell.set_stroke(WHITE, width=1)
                
                if j == 0:
                    cell.next_to(table[-1][0], DOWN, buff=0)
                else:
                    cell.next_to(row[-1], RIGHT, buff=0)
                
                # Color the gene names
                if j == 0:
                    cell_text = Text(text, font_size=20, color=GENE_COLOR)
                else:
                    cell_text = Text(text, font_size=18)
                
                cell_text.move_to(cell.get_center())
                
                row.add(VGroup(cell, cell_text))
            
            table.add(row)
        
        # Add interpretation
        interpretation = Text(
            "The model identified key methylation differences between ME/CFS and Long COVID",
            font_size=24
        )
        interpretation.next_to(table, DOWN, buff=0.5)
        
        return VGroup(table, interpretation)
    
    def create_attention_map(self):
        """Create a visualization of attention maps."""
        # Create a heatmap-like grid for attention visualization
        grid_size = 8
        cell_size = 0.3
        
        # Generate some sample attention patterns
        # Higher number = stronger attention
        attention_data = np.zeros((grid_size, grid_size))
        
        # Pattern 1: Focused attention on specific regions (for head 1)
        attention_data[1:3, 2:4] = np.random.uniform(0.7, 0.9, (2, 2))
        
        # Pattern 2: Diffuse attention (for head 5)
        attention_data[3:5, :] = np.random.uniform(0.3, 0.5, (2, grid_size))
        
        # Pattern 3: Some strong connections between distant regions (for head 2)
        attention_data[6, 1] = 0.8
        attention_data[6, 5] = 0.9
        attention_data[2, 7] = 0.85
        
        # Create the attention map grid
        attention_grid = VGroup()
        
        for i in range(grid_size):
            for j in range(grid_size):
                opacity = attention_data[i, j]
                cell = Square(side_length=cell_size)
                cell.set_fill(YELLOW, opacity=opacity)
                cell.set_stroke(WHITE, width=0.5)
                cell.move_to([j*cell_size, -i*cell_size, 0])
                
                attention_grid.add(cell)
        
        # Add labels for attention heads
        head_labels = VGroup()
        for i in range(grid_size):
            label = Text(f"Head {i+1}", font_size=16)
            label.next_to(attention_grid[i*grid_size], LEFT, buff=0.3)
            head_labels.add(label)
        
        # Add feature indices
        feature_labels = VGroup()
        for j in range(grid_size):
            label = Text(f"F{j+1}", font_size=16)
            label.next_to(attention_grid[j], UP, buff=0.3)
            feature_labels.add(label)
        
        # Add explanations for specific patterns
        explanations = VGroup()
        
        explanation1 = Text("Head 2: Focuses on immune gene clusters", font_size=20, color=GREEN)
        explanation1.to_edge(RIGHT, buff=1)
        explanation1.shift(UP * 1.5)
        
        explanation2 = Text("Head 5: Broad scanning for anomalies", font_size=20, color=BLUE)
        explanation2.next_to(explanation1, DOWN, buff=0.5)
        
        explanation3 = Text("Head 7: Stress-response gene focus", font_size=20, color=RED)
        explanation3.next_to(explanation2, DOWN, buff=0.5)
        
        explanations.add(explanation1, explanation2, explanation3)
        
        # Connect explanations to relevant parts of the grid
        arrows = VGroup()
        
        arrow1 = Arrow(explanation1.get_left(), attention_grid[6*grid_size + 1].get_center(), buff=0.1)
        arrow2 = Arrow(explanation2.get_left(), attention_grid[4*grid_size + 3].get_center(), buff=0.1)
        arrow3 = Arrow(explanation3.get_left(), attention_grid[7*grid_size + 5].get_center(), buff=0.1)
        
        arrows.add(arrow1, arrow2, arrow3)
        
        # Add title
        title = Text("Attention Map for ME/CFS Sample", font_size=24)
        title.next_to(feature_labels, UP, buff=0.5)
        
        return VGroup(attention_grid, head_labels, feature_labels, explanations, arrows, title)
    
    def create_conclusions(self):
        """Create a visualization summarizing the conclusions."""
        # Create a list of conclusions
        conclusions = VGroup(
            Text("1. Transformer architecture effectively models methylation patterns", font_size=28),
            Text("2. 97.06% accuracy in distinguishing ME/CFS, Long COVID, and controls", font_size=28),
            Text("3. Identified distinct epigenetic signatures for each condition", font_size=28),
            Text("4. Interpretable results provide mechanistic insights", font_size=28),
            Text("5. Correlation with clinical metrics suggests potential diagnostic use", font_size=28)
        )
        
        conclusions.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        # Highlight key points
        highlights = VGroup()
        
        keywords = [
            ["Transformer"],
            ["97.06%", "accuracy"],
            ["distinct", "epigenetic signatures"],
            ["Interpretable", "mechanistic insights"],
            ["diagnostic"]
        ]
        
        for i, txt in enumerate(conclusions):
            for keyword in keywords[i]:
                # Find the keyword in the text
                txt_str = txt.text
                start_idx = txt_str.find(keyword)
                if start_idx != -1:
                    # Create a highlight rectangle
                    highlight = Rectangle(
                        width=len(keyword) * 0.18,  # Approximate width per character
                        height=0.3,
                        color=YELLOW,
                        fill_opacity=0.3
                    )
                    
                    # Find the position of the submobject
                    # This is an approximation, in actual code you'd need to locate the exact submobjects
                    highlight.move_to(txt.get_center())
                    highlight.shift(RIGHT * (start_idx - len(txt_str)/2) * 0.1)
                    
                    highlights.add(highlight)
        
        # Add a visual summary
        summary_rect = Rectangle(width=8, height=1.5, color=GREEN)
        summary_rect.next_to(conclusions, DOWN, buff=0.8)
        
        summary_text = Text(
            "This work demonstrates how advanced mathematics can extract\n" +
            "meaningful biological patterns from complex epigenetic data",
            font_size=24,
            line_spacing=1.2
        )
        summary_text.move_to(summary_rect.get_center())
        
        return VGroup(conclusions, highlights, summary_rect, summary_text)

class MathematicalInsights(Scene):
    """Connecting the mathematics to biological insights."""
    
    def construct(self):
        # Title
        title = Text("From Mathematical Models to Biological Insights", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Show the mathematical pipeline
        pipeline_title = Text("Mathematical Pipeline:", font_size=36)
        pipeline_title.next_to(title, DOWN, buff=0.8).to_edge(LEFT)
        
        self.play(Write(pipeline_title))
        
        pipeline = self.create_math_pipeline()
        pipeline.next_to(pipeline_title, DOWN, buff=0.5)
        
        self.play(Create(pipeline, run_time=3))
        self.wait()
        
        # Key mathematical insights
        insights_title = Text("Key Mathematical Insights:", font_size=36)
        insights_title.next_to(pipeline, DOWN, buff=0.8)
        
        self.play(Write(insights_title))
        
        insights = self.create_math_insights()
        insights.next_to(insights_title, DOWN, buff=0.5)
        
        self.play(Create(insights, run_time=2))
        self.wait(2)
        
        # Biological interpretation
        self.play(
            FadeOut(pipeline_title),
            FadeOut(pipeline),
            FadeOut(insights_title),
            FadeOut(insights)
        )
        
        bio_title = Text("Biological Interpretation:", font_size=36)
        bio_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(bio_title))
        
        bio_interpretation = self.create_bio_interpretation()
        bio_interpretation.next_to(bio_title, DOWN, buff=0.5)
        
        self.play(Create(bio_interpretation, run_time=2))
        self.wait(2)
        
        # Final thoughts
        final_title = Text("The Power of Mathematical Modeling:", font_size=36)
        final_title.next_to(bio_interpretation, DOWN, buff=0.8)
        
        self.play(Write(final_title))
        
        final_thoughts = self.create_final_thoughts()
        final_thoughts.next_to(final_title, DOWN, buff=0.5)
        
        self.play(Write(final_thoughts))
        self.wait(2)
        
        self.play(
            FadeOut(title),
            FadeOut(bio_title),
            FadeOut(bio_interpretation),
            FadeOut(final_title),
            FadeOut(final_thoughts),
            run_time=1.5
        )
    
    def create_math_pipeline(self):
        """Create a visualization of the mathematical pipeline."""
        # Create a flowchart of mathematical steps
        flowchart = VGroup()
        
        # Blocks
        block1 = Rectangle(width=4, height=0.8, color=BLUE)
        block1_text = Text("Methylation Data Embedding", font_size=20)
        block1_text.move_to(block1.get_center())
        block1_group = VGroup(block1, block1_text)
        
        block2 = Rectangle(width=4, height=0.8, color=ATTENTION_COLOR)
        block2_text = Text("Self-Attention Mechanism", font_size=20)
        block2_text.move_to(block2.get_center())
        block2_group = VGroup(block2, block2_text)
        block2_group.next_to(block1_group, DOWN, buff=0.5)
        
        block3 = Rectangle(width=4, height=0.8, color=EXPERT_COLOR)
        block3_text = Text("Mixture-of-Experts FFN", font_size=20)
        block3_text.move_to(block3.get_center())
        block3_group = VGroup(block3, block3_text)
        block3_group.next_to(block2_group, DOWN, buff=0.5)
        
        block4 = Rectangle(width=4, height=0.8, color=RED)
        block4_text = Text("Adaptive Computation Time", font_size=20)
        block4_text.move_to(block4.get_center())
        block4_group = VGroup(block4, block4_text)
        block4_group.next_to(block3_group, DOWN, buff=0.5)
        
        block5 = Rectangle(width=4, height=0.8, color=GREEN)
        block5_text = Text("Classification Output", font_size=20)
        block5_text.move_to(block5.get_center())
        block5_group = VGroup(block5, block5_text)
        block5_group.next_to(block4_group, DOWN, buff=0.5)
        
        # Connect with arrows
        arrows = VGroup()
        for i in range(4):
            block_pairs = [
                (block1_group, block2_group),
                (block2_group, block3_group),
                (block3_group, block4_group),
                (block4_group, block5_group)
            ]
            arrow = Arrow(block_pairs[i][0].get_bottom(), block_pairs[i][1].get_top(), buff=0.1)
            arrows.add(arrow)
        
        # Add mathematical formulas next to blocks
        formulas = VGroup()
        
        formula1 = MathTex(r"Z^0 = E + P", font_size=24)
        formula1.next_to(block1_group, RIGHT, buff=1)
        
        formula2 = MathTex(r"A = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)", font_size=24)
        formula2.next_to(block2_group, RIGHT, buff=1)
        
        formula3 = MathTex(r"\text{MoE-FFN}(x) = \sum_{e=1}^{4} g_e(x) \cdot \text{FFN}_e(x)", font_size=24)
        formula3.next_to(block3_group, RIGHT, buff=1)
        
        formula4 = MathTex(r"p = \sigma(w^T x_{\text{mean}})", font_size=24)
        formula4.next_to(block4_group, RIGHT, buff=1)
        
        formula5 = MathTex(r"y = \text{softmax}(W_{\text{cls}_2}\text{GELU}(W_{\text{cls}_1} x_{\text{mean}}))", font_size=22)
        formula5.next_to(block5_group, RIGHT, buff=1)
        
        formulas.add(formula1, formula2, formula3, formula4, formula5)
        
        flowchart.add(block1_group, block2_group, block3_group, block4_group, block5_group, arrows, formulas)
        
        return flowchart
    
    def create_math_insights(self):
        """Create a visualization highlighting key mathematical insights."""
        # Create a list of insights
        insights = VGroup(
            Text("1. Self-attention creates a similarity matrix between CpG sites", font_size=24),
            Text("2. Multi-head attention allows parallel focus on diverse patterns", font_size=24),
            Text("3. Mixture-of-Experts enables specialization in different pathways", font_size=24),
            Text("4. Adaptive Computation allocates more processing to ambiguous cases", font_size=24)
        )
        
        insights.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        
        # Add mathematical representations next to each insight
        math_viz = VGroup()
        
        # 1. Self-attention similarity matrix
        matrix_grid = VGroup()
        for i in range(4):
            for j in range(4):
                # Higher similarity for nearby positions
                opacity = 1 - 0.2 * abs(i - j)
                cell = Square(side_length=0.3)
                cell.set_fill(YELLOW, opacity=opacity)
                cell.set_stroke(WHITE, width=0.5)
                cell.move_to([j*0.3, -i*0.3, 0])
                matrix_grid.add(cell)
        
        matrix_grid.next_to(insights[0], RIGHT, buff=1)
        math_viz.add(matrix_grid)
        
        # 2. Multi-head attention
        heads = VGroup()
        for i in range(2):
            for j in range(4):
                head = Circle(radius=0.15)
                head.set_fill(BLUE, opacity=0.8)
                head.set_stroke(WHITE, width=0.5)
                head.move_to([j*0.4, -i*0.4, 0])
                heads.add(head)
        
        heads.next_to(insights[1], RIGHT, buff=1)
        math_viz.add(heads)
        
        # 3. Mixture-of-Experts
        experts = VGroup()
        for i in range(4):
            expert = Rectangle(width=0.3, height=0.6)
            expert.set_fill(EXPERT_COLOR, opacity=0.8)
            expert.set_stroke(WHITE, width=0.5)
            expert.move_to([i*0.4, 0, 0])
            experts.add(expert)
        
        gate = Triangle().scale(0.2)
        gate.set_fill(RED, opacity=0.8)
        gate.set_stroke(WHITE, width=0.5)
        gate.next_to(experts, UP, buff=0.3)
        
        moe_group = VGroup(experts, gate)
        moe_group.next_to(insights[2], RIGHT, buff=1)
        math_viz.add(moe_group)
        
        # 4. Adaptive Computation
        act_blocks = VGroup()
        n_blocks = [1, 2, 3]  # Different iterations
        for i, n in enumerate(n_blocks):
            blocks = VGroup()
            for j in range(n):
                block = Rectangle(width=0.3, height=0.3)
                block.set_fill(GREEN, opacity=0.8 - j*0.2)
                block.set_stroke(WHITE, width=0.5)
                block.move_to([j*0.4, 0, 0])
                blocks.add(block)
            
            blocks.next_to(insights[3], RIGHT, buff=1 + i*0.8)
            act_blocks.add(blocks)
        
        math_viz.add(act_blocks)
        
        return VGroup(insights, math_viz)
    
    def create_bio_interpretation(self):
        """Create a visualization linking mathematical models to biological interpretation."""
        # Create two columns: Math Model vs Biological Meaning
        table = VGroup()
        
        # Headers
        header1 = Rectangle(width=5, height=0.8, color=BLUE)
        header1_text = Text("Mathematical Component", font_size=24)
        header1_text.move_to(header1.get_center())
        header1_group = VGroup(header1, header1_text)
        
        header2 = Rectangle(width=5, height=0.8, color=GREEN)
        header2_text = Text("Biological Interpretation", font_size=24)
        header2_text.move_to(header2.get_center())
        header2_group = VGroup(header2, header2_text)
        header2_group.next_to(header1_group, RIGHT, buff=0)
        
        table.add(VGroup(header1_group, header2_group))
        
        # Row data
        math_components = [
            "Self-Attention Weights",
            "Multi-Head Attention",
            "Mixture-of-Experts",
            "Adaptive Computation"
        ]
        
        bio_interpretations = [
            "Correlation between CpG sites in regulatory regions",
            "Parallel processing of different gene pathways",
            "Specialized handling of immune vs metabolic patterns",
            "More analysis for borderline cases between conditions"
        ]
        
        # Create rows
        for i in range(len(math_components)):
            # Math component cell
            cell1 = Rectangle(width=5, height=0.8)
            cell1.set_stroke(WHITE, width=1)
            cell1.next_to(table[-1][0], DOWN, buff=0)
            
            cell1_text = Text(math_components[i], font_size=20)
            cell1_text.move_to(cell1.get_center())
            cell1_group = VGroup(cell1, cell1_text)
            
            # Bio interpretation cell
            cell2 = Rectangle(width=5, height=0.8)
            cell2.set_stroke(WHITE, width=1)
            cell2.next_to(cell1, RIGHT, buff=0)
            
            cell2_text = Text(bio_interpretations[i], font_size=18)
            cell2_text.move_to(cell2.get_center())
            cell2_group = VGroup(cell2, cell2_text)
            
            table.add(VGroup(cell1_group, cell2_group))
        
        # Add visualization of a specific example
        example_title = Text("Example: Attention to NR3C1 Gene", font_size=28)
        example_title.next_to(table, DOWN, buff=0.8)
        
        # Create a simple visualization of attention to a gene
        gene_viz = Rectangle(width=6, height=1, color=GENE_COLOR, fill_opacity=0.2)
        gene_viz.next_to(example_title, DOWN, buff=0.4)
        
        gene_label = Text("NR3C1 Gene (Stress Response)", font_size=20, color=GENE_COLOR)
        gene_label.move_to(gene_viz.get_center())
        
        # Add CpG sites with attention
        cpg_sites = VGroup()
        attentions = [0.3, 0.8, 0.5, 0.9, 0.4]
        
        for i, att in enumerate(attentions):
            cpg = Circle(radius=0.2)
            cpg.set_fill(ATTENTION_COLOR, opacity=att)
            cpg.set_stroke(WHITE, width=1)
            cpg.move_to(gene_viz.get_center() + RIGHT * (i - 2) * 0.8)
            
            label = Text(f"CpG {i+1}", font_size=16)
            label.next_to(cpg, DOWN, buff=0.2)
            
            att_label = Text(f"{att:.1f}", font_size=16, color=ATTENTION_COLOR)
            att_label.next_to(cpg, UP, buff=0.2)
            
            cpg_sites.add(VGroup(cpg, label, att_label))
        
        explanation = Text(
            "The model pays highest attention to CpG sites in the promoter region,\n" +
            "which control gene expression of this stress hormone receptor",
            font_size=20,
            line_spacing=1
        )
        explanation.next_to(cpg_sites, DOWN, buff=0.5)
        
        return VGroup(table, example_title, gene_viz, gene_label, cpg_sites, explanation)
    
    def create_final_thoughts(self):
        """Create a visualization for final thoughts."""
        final_text = Text(
            "The transformer architecture creates a mathematical 'bridge'\n" +
            "between complex epigenetic patterns and clinical classifications,\n" +
            "demonstrating how advanced mathematics can unlock biological insights\n" +
            "that traditional statistical methods might miss.",
            font_size=28,
            line_spacing=1.2
        )
        
        return final_text

class Conclusion(Scene):
    """Final conclusion and takeaways."""
    
    def construct(self):
        # Title
        title = Text("Understanding the Epigenomic Transformer", font_size=56)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()
        
        # Key learnings
        learnings_title = Text("Key Mathematical Concepts", font_size=40)
        learnings_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(learnings_title))
        
        # List the key mathematical concepts
        concepts = VGroup(
            Text("1. Self-attention as a measure of feature similarity", font_size=32),
            Text("2. Multi-head mechanisms for parallel pattern recognition", font_size=32),
            Text("3. Mixture-of-Experts for specialized processing", font_size=32),
            Text("4. Adaptive computation for variable-depth processing", font_size=32),
            Text("5. Self-supervised pretraining for representation learning", font_size=32)
        )
        
        concepts.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        concepts.next_to(learnings_title, DOWN, buff=0.5)
        
        for concept in concepts:
            self.play(FadeIn(concept, shift=UP*0.2))
            self.wait(0.3)
        
        self.wait()
        
        # Final thoughts
        self.play(
            FadeOut(learnings_title),
            FadeOut(concepts)
        )
        
        final_title = Text("The Power of Mathematics in Medicine", font_size=44)
        final_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(final_title))
        
        final_message = Text(
            "By modeling complex epigenetic patterns with transformer architectures,\n" +
            "we can achieve clinical-grade classification accuracy while gaining\n" +
            "interpretable insights into disease mechanisms. This approach\n" +
            "demonstrates how mathematical innovation drives medical discovery.",
            font_size=32,
            line_spacing=1.2
        )
        final_message.next_to(final_title, DOWN, buff=0.5)
        
        self.play(Write(final_message, run_time=2))
        self.wait(2)
        
        # 3Blue1Brown-style ending
        ending = Text("Thanks for watching", font_size=48)
        ending.to_edge(DOWN, buff=1)
        
        self.play(Write(ending))
        self.wait(3)
        
        self.play(
            FadeOut(title),
            FadeOut(final_title),
            FadeOut(final_message),
            FadeOut(ending),
            run_time=1.5
        )

# Main function to run the animation
if __name__ == "__main__":
    # Uncomment the scene you want to render
    # Note: For a full video, you would render all scenes and combine them
    
    # Introduction
    # os.system("manim -pqh manim_script.py Introduction")
    
    # Data representation
    # os.system("manim -pqh manim_script.py DataRepresentation")
    
    # Transformer overview
    # os.system("manim -pqh manim_script.py TransformerOverview")
    
    # Input embedding
    # os.system("manim -pqh manim_script.py InputEmbedding")
    
    # Self-attention
    # os.system("manim -pqh manim_script.py SelfAttentionMechanism")
    
    # Mixture of experts
    # os.system("manim -pqh manim_script.py MixtureOfExperts")
    
    # Adaptive computation time
    # os.system("manim -pqh manim_script.py AdaptiveComputationTime")
    
    # Masked pretraining
    # os.system("manim -pqh manim_script.py MaskedPretraining")
    
    # Results and conclusion
    # os.system("manim -pqh manim_script.py ResultsAndConclusion")
    
    # Mathematical insights
    # os.system("manim -pqh manim_script.py MathematicalInsights")
    
    # Final conclusion
    # os.system("manim -pqh manim_script.py Conclusion")
    
    # For rendering all scenes
    scenes = [
        "Introduction",
        "DataRepresentation",
        "TransformerOverview",
        "InputEmbedding",
        "SelfAttentionMechanism",
        "MixtureOfExperts",
        "AdaptiveComputationTime",
        "MaskedPretraining",
        "ResultsAndConclusion",
        "MathematicalInsights",
        "Conclusion"
    ]
    
    # Generate a full command to render all scenes
    full_command = "manim -pqh manim_script.py " + " ".join(scenes)
    print(f"To render all scenes, run: {full_command}")
    
    # Alternatively, render a specific scene
    # Choose which scene to render by uncommenting one of these:
    scene_to_render = "SelfAttentionMechanism"  # Example: just render self-attention
    # scene_to_render = "MixtureOfExperts"
    # scene_to_render = "AdaptiveComputationTime"
    
    render_command = f"manim -pqh {__file__} {scene_to_render}"
    print(f"Rendering scene: {scene_to_render}")
    print(f"Command: {render_command}")
    
    # Execute the command
    import subprocess
    subprocess.run(render_command, shell=True)