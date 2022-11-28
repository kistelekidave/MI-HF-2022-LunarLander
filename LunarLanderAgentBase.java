import java.util.Arrays;

public class LunarLanderAgentBase {
	
	static int tXMax = 40;
	static int tYMax = 40;
	static int vXMax = 21;
	static int vYMax = 21;
	
    // The resolution of the observation space
    // The four variables of the observation space, from left to right:
    //   0: X component of the vector pointing to the middle of the platform from the lander
    //   1: Y component of the vector pointing to the middle of the platform from the lander
    //   2: X component of the velocity vector of the lander
    //   3: Y component of the velocity vector of the lander
    static final int[] OBSERVATION_SPACE_RESOLUTION = {tXMax + 1, tYMax + 1, vXMax + 1, vYMax + 1};

    final double[][] observationSpace;
    double[][][][][] qTable;
    final int[] envActionSpace;
    private final int nIterations;

    double epsilon = 1.0;
    int iteration = 0;
    boolean test = false;

    // your variables here
    // ...
    double[][][][][] bestTable;
    double bestReward = -200;
    double lastReward = -200;

    double alpha = 0.1;
    double gamma = 0.6;
    int epsilon_step = 100;
    double epsilon_decay = 0.99;
    int save_interval = 1000;
    
    int epoch = 0;

    public LunarLanderAgentBase(double[][] observationSpace, int[] actionSpace, int nIterations) {
        this.observationSpace = observationSpace;
        this.qTable =
                new double[OBSERVATION_SPACE_RESOLUTION[0]]
                        [OBSERVATION_SPACE_RESOLUTION[1]]
                        [OBSERVATION_SPACE_RESOLUTION[2]]
                        [OBSERVATION_SPACE_RESOLUTION[3]]
                        [actionSpace.length];
        this.envActionSpace = actionSpace;
        this.nIterations = nIterations;
    }

    public static int[] quantizeState(double[][] observationSpace, double[] state) {
    	int[] res = new int[4];
    	
    	//res[0] = (int) ((state[0] + 300) / 10);
    	res[0] = (int) (Math.pow(state[0], (3.0 / 5.0)) * (tXMax / 2 / 30.63887) + tXMax / 2);
    	if (state[0] > -15 && state[0] < 15)
    		res[0] = 20;
    	
        //res[1] = (int) ((state[1]) / 4);
        res[1] = (int) (Math.log(state[1] / 100 + 1) * tYMax / 1.098612);
        if (state[1] < 0)
        	res[1] = 0;
        else if (state[1] > 180)
        	res[1] = tYMax;
        
        //res[2] = (int) ((state[2] + 7) * vXMax / 14);
    	res[2] = (int) (Math.pow(state[2], (3.0 / 5.0)) * (vXMax / 2 / 3.21409585) + vXMax / 2);

        //res[3] = (int) ((state[3] + 7) * vYMax / 14);
        res[3] = (int) (Math.pow(state[3], (3.0 / 5.0)) * (vYMax / 2 / 3.21409585) + vYMax / 2);
        
        return res;
    }

    public void epochEnd(double epochRewardSum) {
    	return;
    }

    public void learn(double[] oldState, int action, double[] newState, double reward) {

        int[] quantizedOldState = quantizeState(observationSpace, oldState);
        
        double newReward = (1 - alpha) * oldState[action] + alpha * (reward + gamma * maxQ(newState));

        qTable[quantizedOldState[0]][quantizedOldState[1]][quantizedOldState[2]][quantizedOldState[3]][action] = newReward;
    }

    private double maxQ(double[] state) {
    	
    	int[] quantized = quantizeState(observationSpace, state);
    	
    	double max = qTable[quantized[0]][quantized[1]][quantized[2]][quantized[3]][0];
		for (int i = 1; i < state.length; i++) {
			if (qTable[quantized[0]][quantized[1]][quantized[2]][quantized[3]][i] > max)
				max = qTable[quantized[0]][quantized[1]][quantized[2]][quantized[3]][i];
		}
		return max;
	}

	public void trainEnd() {
        test = true;
    }
}
