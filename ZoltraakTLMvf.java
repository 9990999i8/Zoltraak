import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

public class ZoltraakTLMvf {

    // ---------------------------------------------------------------------------------
    // Configuration
    // ---------------------------------------------------------------------------------
    static class Config {
        int dim;
        int hidden_dim; // d_ff
        int n_layers;
        int n_heads;
        int vocab_size;
        int max_seq_len;
    }
    
    // ---------------------------------------------------------------------------------
    // Tensor Math (Minimal implementation for CPU inference)
    // ---------------------------------------------------------------------------------
    static class Tensor {
        float[] data;
        int[] shape;

        public Tensor(int... shape) {
            this.shape = shape;
            int size = 1;
            for (int s : shape) size *= s;
            this.data = new float[size];
        }

        public Tensor(float[] data, int... shape) {
            this.data = data;
            this.shape = shape;
        }

        // Basic matmul: (M, K) @ (K, N) -> (M, N)
        public static Tensor matmul(Tensor A, Tensor B) {
            int aDim = A.shape.length;
            int bDim = B.shape.length;
            
            // Normalize dims to at least 2
            if (aDim < 2 || bDim < 2) throw new IllegalArgumentException("Matmul requires at least 2D tensors");
            
            int M = A.shape[aDim-2];
            int K = A.shape[aDim-1];
            
            if (B.shape[bDim-2] != K) throw new IllegalArgumentException("Matmul dim mismatch: " + K + " vs " + B.shape[bDim-2]);
            
            int N = B.shape[bDim-1];
            
            // Simple handling: If dimensions don't match exactly (e.g. 3D vs 2D), we broadcast/flatten.
            // My inference code uses (1, T, C) @ (C, Hs).
            // A is 3D (1, T, C). B is 2D (C, Hs).
            // Result should be (1, T, Hs).
            
            int batchA = (aDim == 3) ? A.shape[0] : 1;
            int batchB = (bDim == 3) ? B.shape[0] : 1;
            
            // Check broadcasting
            if (batchA != batchB && batchA != 1 && batchB != 1) {
                throw new UnsupportedOperationException("Batch size mismatch and no broadcast: " + batchA + " vs " + batchB);
            }
            int outBatch = Math.max(batchA, batchB);
            
            Tensor C;
            if (aDim == 3 || bDim == 3) {
                C = new Tensor(outBatch, M, N);
            } else {
                C = new Tensor(M, N);
            }
            
            for (int b = 0; b < outBatch; b++) {
                // Determine offsets
                // If A has batch 1, we reuse same data for all output batches (broadcast)
                int offA = (batchA > 1) ? b * M * K : 0;
                int offB = (batchB > 1) ? b * K * N : 0;
                int offC = b * M * N;
                
                matmul2D(A.data, offA, B.data, offB, C.data, offC, M, K, N);
            }
            return C;
        }

        private static void matmul2D(float[] A, int aOff, float[] B, int bOff, float[] C, int cOff, int M, int K, int N) {
            for (int i = 0; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    float valA = A[aOff + i * K + k];
                    for (int j = 0; j < N; j++) {
                        C[cOff + i * N + j] += valA * B[bOff + k * N + j];
                    }
                }
            }
        }
        
        public Tensor transpose() {
            int rank = shape.length;
            int M = shape[rank-2];
            int N = shape[rank-1];
            
            int[] newShape = shape.clone();
            newShape[rank-2] = N;
            newShape[rank-1] = M;
            
            Tensor out = new Tensor(newShape);
            
            if (rank == 2) {
                for(int i=0; i<M; i++)
                    for(int j=0; j<N; j++)
                        out.data[j*M + i] = this.data[i*N + j];
            } else if (rank == 3) {
                int B = shape[0];
                for(int b=0; b<B; b++) {
                    int offIn = b*M*N;
                    int offOut = b*N*M;
                    for(int i=0; i<M; i++)
                        for(int j=0; j<N; j++)
                            out.data[offOut + j*M + i] = this.data[offIn + i*N + j];
                }
            }
            return out;
        }

        public Tensor add(Tensor other) {
            if (this.data.length != other.data.length) {
                if (other.data.length == this.shape[this.shape.length-1]) {
                    Tensor out = new Tensor(this.shape);
                    int lastDim = other.data.length;
                    for (int i = 0; i < this.data.length; i++) {
                        out.data[i] = this.data[i] + other.data[i % lastDim];
                    }
                    return out;
                }
                throw new IllegalArgumentException("Add shape mismatch");
            }
            Tensor out = new Tensor(this.shape);
            for (int i = 0; i < this.data.length; i++) out.data[i] = this.data[i] + other.data[i];
            return out;
        }

        public void softmax() {
            int lastDim = shape[shape.length-1];
            int numRows = data.length / lastDim;
            
            for (int i = 0; i < numRows; i++) {
                int offset = i * lastDim;
                float max = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < lastDim; j++) {
                    if (data[offset+j] > max) max = data[offset+j];
                }
                
                float sum = 0;
                for (int j = 0; j < lastDim; j++) {
                    float e = (float) Math.exp(data[offset+j] - max);
                    data[offset+j] = e;
                    sum += e;
                }
                
                for (int j = 0; j < lastDim; j++) {
                    data[offset+j] /= sum;
                }
            }
        }
        
        public Tensor layerNorm(Tensor gamma, Tensor beta) {
            int lastDim = shape[shape.length-1];
            int numRows = data.length / lastDim;
            Tensor out = new Tensor(shape);
            
            for(int i=0; i<numRows; i++) {
                int offset = i*lastDim;
                float mean = 0;
                for(int j=0; j<lastDim; j++) mean += data[offset+j];
                mean /= lastDim;
                
                float var = 0;
                for(int j=0; j<lastDim; j++) var += (data[offset+j]-mean)*(data[offset+j]-mean);
                var /= lastDim;
                
                float std = (float) Math.sqrt(var + 1e-5f);
                
                for(int j=0; j<lastDim; j++) {
                    float norm = (data[offset+j] - mean) / std;
                    out.data[offset+j] = norm * gamma.data[j] + beta.data[j];
                }
            }
            return out;
        }
        
        public Tensor relu() {
            Tensor out = new Tensor(shape);
            for(int i=0; i<data.length; i++) {
                out.data[i] = data[i] > 0 ? data[i] : 0;
            }
            return out;
        }
        
        public static Tensor embeddingLookup(Tensor weight, int[] ids) {
            int dim = weight.shape[1];
            Tensor out = new Tensor(1, ids.length, dim);
            for (int t = 0; t < ids.length; t++) {
                int id = ids[t];
                System.arraycopy(weight.data, id * dim, out.data, t * dim, dim);
            }
            return out;
        }
    }

    // ---------------------------------------------------------------------------------
    // Tokenizer
    // ---------------------------------------------------------------------------------
    static class Tokenizer {
        Map<Integer, byte[]> vocab = new HashMap<>();
        Map<String, Integer> merges = new HashMap<>();
        int vocabSize;

        public void load(String path) throws IOException {
            try (DataInputStream dis = new DataInputStream(new FileInputStream(path))) {
                vocabSize = dis.readInt();
                for (int i = 0; i < vocabSize; i++) {
                    int len = dis.readInt();
                    byte[] b = new byte[len];
                    dis.readFully(b);
                    vocab.put(i, b);
                }
                
                if (dis.available() > 0) {
                    int numMerges = dis.readInt();
                    for (int i = 0; i < numMerges; i++) {
                        int p0 = dis.readInt();
                        int p1 = dis.readInt();
                        int idx = dis.readInt();
                        merges.put(p0 + "," + p1, idx);
                    }
                }
            }
        }

        public int[] encode(String text) {
            byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
            List<Integer> ids = new ArrayList<>();
            for (byte b : bytes) ids.add(Byte.toUnsignedInt(b));

            while (ids.size() >= 2) {
                int minMergeId = Integer.MAX_VALUE;
                int p0_best = -1, p1_best = -1;
                
                for(int i=0; i<ids.size()-1; i++) {
                    String key = ids.get(i) + "," + ids.get(i+1);
                    Integer mid = merges.get(key);
                    if (mid != null && mid < minMergeId) {
                        minMergeId = mid;
                        p0_best = ids.get(i);
                        p1_best = ids.get(i+1);
                    }
                }
                
                if (minMergeId == Integer.MAX_VALUE) break;
                
                List<Integer> newIds = new ArrayList<>();
                int i = 0;
                while(i < ids.size()) {
                    if (i < ids.size() - 1 && ids.get(i) == p0_best && ids.get(i+1) == p1_best) {
                        newIds.add(minMergeId);
                        i += 2;
                    } else {
                        newIds.add(ids.get(i));
                        i += 1;
                    }
                }
                ids = newIds;
            }

            return ids.stream().mapToInt(Integer::intValue).toArray();
        }

        public String decode(int[] ids) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            for (int id : ids) {
                byte[] b = vocab.get(id);
                if (b != null) {
                    try { bos.write(b); } catch (IOException e) {}
                }
            }
            return new String(bos.toByteArray(), StandardCharsets.UTF_8);
        }
    }

    // ---------------------------------------------------------------------------------
    // Model Components
    // ---------------------------------------------------------------------------------
    
    static class GPT {
        Config config = new Config();
        Tensor tokenEmb;
        Tensor posEmb;
        
        static class Block {
            Tensor[] q_w, k_w, v_w; 
            Tensor proj_w, proj_b;
            Tensor ln1_w, ln1_b;
            Tensor l1_w, l1_b;
            Tensor l2_w, l2_b;
            Tensor ln2_w, ln2_b;
        }
        
        Block[] blocks;
        Tensor lnf_w, lnf_b;
        Tensor head_w, head_b;

        public void load(String path) throws IOException {
            try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
                int magic = dis.readInt();
                if (magic != 0x54494E59) throw new IOException("Invalid magic number");
                config.dim = dis.readInt();
                config.hidden_dim = dis.readInt();
                config.n_layers = dis.readInt();
                config.n_heads = dis.readInt();
                config.vocab_size = dis.readInt();
                config.max_seq_len = dis.readInt();
                
                System.out.println("Model Config: Layers=" + config.n_layers + ", Dim=" + config.dim + ", Heads=" + config.n_heads);
                
                tokenEmb = readTensor(dis, "token_emb", config.vocab_size, config.dim);
                posEmb = readTensor(dis, "pos_emb", config.max_seq_len, config.dim);
                
                blocks = new Block[config.n_layers];
                int headSize = config.dim / config.n_heads;
                
                for(int i=0; i<config.n_layers; i++) {
                    blocks[i] = new Block();
                    blocks[i].q_w = new Tensor[config.n_heads];
                    blocks[i].k_w = new Tensor[config.n_heads];
                    blocks[i].v_w = new Tensor[config.n_heads];
                    
                    for(int h=0; h<config.n_heads; h++) blocks[i].q_w[h] = readTensor(dis, "b"+i+"_q"+h, headSize, config.dim).transpose();
                    for(int h=0; h<config.n_heads; h++) blocks[i].k_w[h] = readTensor(dis, "b"+i+"_k"+h, headSize, config.dim).transpose();
                    for(int h=0; h<config.n_heads; h++) blocks[i].v_w[h] = readTensor(dis, "b"+i+"_v"+h, headSize, config.dim).transpose();
                    
                    blocks[i].proj_w = readTensor(dis, "b"+i+"_proj_w", config.dim, config.dim).transpose();
                    blocks[i].proj_b = readTensor(dis, "b"+i+"_proj_b", config.dim);
                    
                    blocks[i].ln1_w = readTensor(dis, "b"+i+"_ln1_w", config.dim);
                    blocks[i].ln1_b = readTensor(dis, "b"+i+"_ln1_b", config.dim);
                    
                    blocks[i].l1_w = readTensor(dis, "b"+i+"_mlp1_w", 4*config.dim, config.dim).transpose();
                    blocks[i].l1_b = readTensor(dis, "b"+i+"_mlp1_b", 4*config.dim);
                    blocks[i].l2_w = readTensor(dis, "b"+i+"_mlp2_w", config.dim, 4*config.dim).transpose();
                    blocks[i].l2_b = readTensor(dis, "b"+i+"_mlp2_b", config.dim);
                    
                    blocks[i].ln2_w = readTensor(dis, "b"+i+"_ln2_w", config.dim);
                    blocks[i].ln2_b = readTensor(dis, "b"+i+"_ln2_b", config.dim);
                }
                
                lnf_w = readTensor(dis, "lnf_w", config.dim);
                lnf_b = readTensor(dis, "lnf_b", config.dim);
                
                head_w = readTensor(dis, "head_w", config.vocab_size, config.dim).transpose();
                head_b = readTensor(dis, "head_b", config.vocab_size);
            }
        }
        
        private Tensor readTensor(DataInputStream dis, String name, int... shape) throws IOException {
            int size = 1;
            for(int s : shape) size *= s;
            System.out.println("Reading " + name + " shape=" + Arrays.toString(shape) + " size=" + size);
            
            byte[] bytes = new byte[size * 4];
            dis.readFully(bytes);
            FloatBuffer fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] data = new float[size];
            fb.get(data);
            return new Tensor(data, shape);
        }

        public Tensor forward(int[] idx) {
            int T = idx.length;
            Tensor x = Tensor.embeddingLookup(tokenEmb, idx);
            
            for(int t=0; t<T; t++) {
                int pos = t; 
                for(int c=0; c<config.dim; c++) {
                     x.data[t*config.dim + c] += posEmb.data[pos*config.dim + c];
                }
            }
            
            for (Block b : blocks) {
                Tensor resid = x;
                Tensor x_ln = x.layerNorm(b.ln1_w, b.ln1_b);
                
                Tensor[] headsOut = new Tensor[config.n_heads];
                int headSize = config.dim / config.n_heads;
                
                for(int h=0; h<config.n_heads; h++) {
                    Tensor q = Tensor.matmul(x_ln, b.q_w[h]); 
                    Tensor k = Tensor.matmul(x_ln, b.k_w[h]); 
                    Tensor v = Tensor.matmul(x_ln, b.v_w[h]);
                    
                    q.shape = new int[]{T, headSize};
                    k.shape = new int[]{T, headSize};
                    v.shape = new int[]{T, headSize};
                    
                    Tensor kT = k.transpose();
                    Tensor att = Tensor.matmul(q, kT); 
                    
                    float scale = (float) (1.0 / Math.sqrt(headSize));
                    for(int i=0; i<att.data.length; i++) att.data[i] *= scale;
                    
                    for(int i=0; i<T; i++) {
                        for(int j=0; j<T; j++) {
                            if (j > i) att.data[i*T + j] = Float.NEGATIVE_INFINITY;
                        }
                    }
                    
                    att.softmax();
                    headsOut[h] = Tensor.matmul(att, v);
                }
                
                Tensor concat = new Tensor(1, T, config.dim);
                for(int t=0; t<T; t++) {
                    for(int h=0; h<config.n_heads; h++) {
                        System.arraycopy(headsOut[h].data, t*headSize, concat.data, t*config.dim + h*headSize, headSize);
                    }
                }
                
                Tensor proj = Tensor.matmul(concat, b.proj_w).add(b.proj_b);
                x = resid.add(proj);
                
                resid = x;
                x_ln = x.layerNorm(b.ln2_w, b.ln2_b);
                
                Tensor l1 = Tensor.matmul(x_ln, b.l1_w).add(b.l1_b);
                Tensor act = l1.relu();
                Tensor l2 = Tensor.matmul(act, b.l2_w).add(b.l2_b);
                
                x = resid.add(l2);
            }
            
            x = x.layerNorm(lnf_w, lnf_b);
            
            Tensor lastTok = new Tensor(1, 1, config.dim);
            System.arraycopy(x.data, (T-1)*config.dim, lastTok.data, 0, config.dim);
            
            Tensor logits = Tensor.matmul(lastTok, head_w).add(head_b);
            return logits;
        }
        
        public int generateNext(int[] idx) {
             Tensor logits = forward(idx);
             logits.softmax();
             float r = (float) Math.random();
             float cdf = 0;
             for(int i=0; i<config.vocab_size; i++) {
                 cdf += logits.data[i];
                 if (r < cdf) return i;
             }
             return config.vocab_size - 1;
        }
    }

    // ---------------------------------------------------------------------------------
    // Main
    // ---------------------------------------------------------------------------------
    public static void main(String[] args) {
        try {
            System.out.println("Loading tokenizer...");
            Tokenizer tokenizer = new Tokenizer();
            tokenizer.load("tokenizer.bin");
            
            System.out.println("Loading model...");
            GPT model = new GPT();
            model.load("model.bin");
            
            System.out.println("Model loaded! Type something (or 'quit' to exit).");
            Scanner scanner = new Scanner(System.in);
            
            while(true) {
                System.out.print("> ");
                String line = scanner.nextLine();
                if (line.equalsIgnoreCase("quit")) break;
                if (line.trim().isEmpty()) continue;
                
                int[] tokens = tokenizer.encode(line);
                
                System.out.print("Replying: " + line);
                for(int i=0; i<100; i++) { 
                    int next = model.generateNext(tokens);
                    System.out.print(tokenizer.decode(new int[]{next}));
                    System.out.flush();
                    
                    int[] newTokens = Arrays.copyOf(tokens, tokens.length + 1);
                    newTokens[tokens.length] = next;
                    tokens = newTokens;
                    
                    if (tokens.length >= model.config.max_seq_len) {
                        tokens = Arrays.copyOfRange(tokens, tokens.length - model.config.max_seq_len + 1, tokens.length);
                    }
                }
                System.out.println("\n");
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
