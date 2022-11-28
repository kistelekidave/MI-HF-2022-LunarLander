import java.util.Arrays;
import java.util.HashMap;

import static java.lang.Math.min;

public class LunarLanderEvaluator {
	public static void main(String[] args) {
        double sum = 0;
        int count1 = 0;
        final int N = 20;
        final double[] res = new double[N];
        final int[] count = new int[N];
        /*IntStream.range(0, N).parallel().forEach(i -> {
            double resb = run(args);
            res[i] = resb;
            if (resb*12 >= 8) {
                count[i] = 1;
            }
        });*/

        for (int i = 0; i < N; i++) {
            double resb = run(args);
            res[i] = resb;
            if (resb*12 >= 10) {
                count[i] = 1;
            }
        }

        sum = Arrays.stream(res).sum();
        count1 = Arrays.stream(count).sum();

        double avg = sum / N;
        double max = Arrays.stream(res).max().getAsDouble();

        System.out.println("Avg: " + (avg*12));
        System.out.println("Count: " + count1);
        System.out.println("Max: " + (max*12));
    }
	
    public static double run(String[] args) {
        int nIterations = (int) 1e6;
        int iteration = 0;

        int[][] randomVelocityRange = {{-1, 1}, {1, 5}};
        Environment env = new Environment(randomVelocityRange);

        LunarLanderAgent agent = new LunarLanderAgent(env.observationSpace,
                env.actionSpace, nIterations);

        while (iteration < nIterations) {
            double[] state = env.reset();

//            int epochIteration = 0;
            double epochRewardSum = 0;
            boolean done = false;

            while (!done) {
                int action = agent.step(state);
                Environment.EnvStepDTO envStep = env.step(action);
                done = env.done;
//                System.out.println(
//                        "step: " + env.stepCounter +
//                        ", state: " + Arrays.toString(envStep.state) +
//                        ", reward:" + envStep.reward
//                );
                agent.learn(state, action, envStep.state, envStep.reward);

                state = envStep.state;

//                epochIteration += 1;
                epochRewardSum += envStep.reward;

                iteration += 1;

            }

            agent.epochEnd(epochRewardSum);
        }

        agent.trainEnd();

        int nTestIterations = 10;
        double rewardSum = 0;

        HashMap<Result, Integer> iterationOutcomes = new HashMap<>();
        for (Result resType : Result.values()) {
            iterationOutcomes.put(resType, 0);
        }

        for (int i = 0; i < nTestIterations; i++) {
            double[] state = env.reset();
            boolean done = false;

            while (!done) {
                int action = agent.step(state);
                Environment.EnvStepDTO envStep = env.step(action);
                state = envStep.state;
                done = envStep.done;

                rewardSum += envStep.reward;
            }

            iterationOutcomes.put(env.result, iterationOutcomes.get(env.result) + 1);
        }

        HashMap<Result, Integer> pointDict = new HashMap<>() {{
            put(Result.LANDED, 4);
            put(Result.LANDING_GEAR_CRASHED, 2);
            put(Result.CRASH_LANDING, 1);
            put(Result.CRASH, 0);
            put(Result.OUT_OF_TIME, 0);
        }};

        int maxPoints = 12;
        int earnedPoints = 0;

        for (Result resType : Result.values()) {
            Integer nOccurrence = iterationOutcomes.get(resType);
            earnedPoints += nOccurrence * pointDict.get(resType);
        }

        double pointFraction = min((double) earnedPoints / (double) maxPoints, 1.0);

        System.out.println("{\"fraction\": " + pointFraction + "}");
        return pointFraction;
    }
    
}
