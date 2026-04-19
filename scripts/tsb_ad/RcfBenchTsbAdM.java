// AWS randomcutforest-java on the TSB-AD multivariate track.
// Same protocol as tests/tsb_ad_m.rs: per-dim z-score on the
// upstream `tr_<N>` train split, frozen-baseline scoring, EMA
// smoothing, point-wise ROC-AUC aggregated per source dataset.
//
// Usage:
//   javac -cp /path/to/randomcutforest-core-4.4.0.jar \
//       scripts/tsb_ad/RcfBenchTsbAdM.java
//   java -cp scripts/tsb_ad:/path/to/...jar \
//       RcfBenchTsbAdM /tmp/tsb-ad/TSB-AD-M

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.amazon.randomcutforest.RandomCutForest;

public class RcfBenchTsbAdM {
    static final int TREES = 100;
    static final int SAMPLE = 256;
    static final double SMOOTH_ALPHA = 0.02;
    static final long SEED = 2026L;
    static final long MIN_POSITIVES = 5;
    static final int DEFAULT_MAX_EVAL = 50000;
    static final Pattern TR_PATTERN = Pattern.compile("_tr_(\\d+)_");

    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.err.println("usage: RcfBenchTsbAdM <tsb-ad-m-dir> [max-eval]");
            System.exit(2);
        }
        File root = new File(args[0]);
        int maxEval = (args.length >= 2) ? Integer.parseInt(args[1]) : DEFAULT_MAX_EVAL;

        File[] csvs = root.listFiles((d, n) -> n.endsWith(".csv"));
        if (csvs == null) {
            System.err.println("no CSVs under " + root);
            System.exit(3);
        }
        Arrays.sort(csvs);

        TreeMap<String, double[]> perDataset = new TreeMap<>();
        long overallPos = 0;
        double overallWeighted = 0.0;
        int overallFiles = 0;
        for (File csv : csvs) {
            Meta meta = parseMeta(csv);
            if (meta == null) continue;
            double[] res = scoreFile(csv, meta, maxEval);
            if (res == null) continue;
            double auc = res[0];
            long pos = (long) res[1];
            int dim = (int) res[2];
            System.out.printf("  %.3f  D=%-3d  pos=%-7d  %-12s  %s%n",
                auc, dim, pos, meta.dataset, csv.getName());
            if (pos >= MIN_POSITIVES) {
                double[] slot = perDataset.computeIfAbsent(meta.dataset, k -> new double[3]);
                slot[0] += auc * pos;
                slot[1] += pos;
                slot[2] += 1;
                overallWeighted += auc * pos;
                overallPos += pos;
                overallFiles += 1;
            }
        }

        System.out.println("\nper-dataset AUC (weighted by positives):");
        for (Map.Entry<String, double[]> e : perDataset.entrySet()) {
            double[] v = e.getValue();
            if (v[1] == 0) continue;
            System.out.printf("  %.3f  files=%-3d  pos=%-7d  %s%n",
                v[0] / v[1], (int) v[2], (long) v[1], e.getKey());
        }
        if (overallPos > 0) {
            System.out.printf(
                "%naggregate weighted AUC: %.3f across %d files / %d positives (AWS Java, max-eval=%d)%n",
                overallWeighted / overallPos, overallFiles, overallPos, maxEval);
        }
    }

    static class Meta {
        String dataset;
        int trainEnd;
    }

    static Meta parseMeta(File csv) {
        Matcher m = TR_PATTERN.matcher(csv.getName());
        if (!m.find()) return null;
        Meta out = new Meta();
        out.trainEnd = Integer.parseInt(m.group(1));
        String stem = csv.getName().replaceFirst("\\.csv$", "");
        String[] parts = stem.split("_");
        out.dataset = (parts.length > 1) ? parts[1] : "unknown";
        return out;
    }

    // Returns [auc, pos, dim] or null on shape mismatch.
    static double[] scoreFile(File csv, Meta meta, int maxEval) throws Exception {
        List<double[]> rows = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        int dim = -1;
        try (BufferedReader br = new BufferedReader(new FileReader(csv))) {
            String header = br.readLine();
            if (header == null) return null;
            dim = header.split(",").length - 1;
            String line;
            while ((line = br.readLine()) != null) {
                if (line.isEmpty()) continue;
                String[] cols = line.split(",");
                if (cols.length != dim + 1) continue;
                double[] row = new double[dim];
                for (int i = 0; i < dim; i++) row[i] = Double.parseDouble(cols[i]);
                rows.add(row);
                labels.add(Double.parseDouble(cols[dim]) >= 0.5 ? 1 : 0);
            }
        }
        int n = rows.size();
        if (n <= meta.trainEnd + 1) return null;

        // Per-dim z-score on the train split.
        double[] mean = new double[dim];
        double[] m2 = new double[dim];
        for (int r = 0; r < meta.trainEnd; r++) {
            double[] row = rows.get(r);
            for (int d = 0; d < dim; d++) mean[d] += row[d];
        }
        for (int d = 0; d < dim; d++) mean[d] /= Math.max(1, meta.trainEnd);
        for (int r = 0; r < meta.trainEnd; r++) {
            double[] row = rows.get(r);
            for (int d = 0; d < dim; d++) {
                double delta = row[d] - mean[d];
                m2[d] += delta * delta;
            }
        }
        double[] std = new double[dim];
        for (int d = 0; d < dim; d++) {
            std[d] = Math.max(Math.sqrt(m2[d] / Math.max(1, meta.trainEnd)), 1e-9);
        }

        RandomCutForest forest = RandomCutForest.builder()
            .dimensions(dim)
            .numberOfTrees(TREES)
            .sampleSize(SAMPLE)
            .randomSeed(SEED)
            .build();

        double[] buf = new double[dim];
        // Warm.
        for (int r = 0; r < meta.trainEnd; r++) {
            double[] row = rows.get(r);
            for (int d = 0; d < dim; d++) buf[d] = (row[d] - mean[d]) / std[d];
            forest.update(buf);
        }

        // Stride-subsample eval rows above maxEval.
        int evalN = n - meta.trainEnd;
        int stride = 1;
        if (evalN > maxEval) stride = Math.max(1, evalN / maxEval);
        int keep = 0;
        for (int r = meta.trainEnd; r < n; r += stride) keep++;

        double[] rawScores = new double[keep];
        int[] evalLabels = new int[keep];
        int out = 0;
        for (int r = meta.trainEnd; r < n && out < keep; r += stride) {
            double[] row = rows.get(r);
            for (int d = 0; d < dim; d++) buf[d] = (row[d] - mean[d]) / std[d];
            rawScores[out] = forest.getAnomalyScore(buf);
            evalLabels[out] = labels.get(r);
            out++;
        }

        // EMA smoothing.
        double[] smoothed = new double[keep];
        if (keep > 0) {
            double acc = rawScores[0];
            for (int i = 0; i < keep; i++) {
                acc = SMOOTH_ALPHA * rawScores[i] + (1 - SMOOTH_ALPHA) * acc;
                smoothed[i] = acc;
            }
        }

        long pos = 0;
        for (int l : evalLabels) if (l == 1) pos++;
        double auc = auc(smoothed, evalLabels);
        return new double[]{auc, pos, dim};
    }

    static double auc(double[] scores, int[] labels) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(scores[b], scores[a]));
        long totalPos = 0;
        for (int l : labels) if (l == 1) totalPos++;
        long totalNeg = labels.length - totalPos;
        if (totalPos == 0 || totalNeg == 0) return 0.5;
        double aucVal = 0.0;
        long tp = 0, fp = 0;
        double prevTpr = 0, prevFpr = 0;
        for (int idx : order) {
            if (labels[idx] == 1) tp++;
            else fp++;
            double tpr = (double) tp / totalPos;
            double fpr = (double) fp / totalNeg;
            aucVal += (fpr - prevFpr) * (tpr + prevTpr) / 2.0;
            prevTpr = tpr;
            prevFpr = fpr;
        }
        return aucVal;
    }
}
