using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;

public enum Team
{
    Blue = 0,
    Purple = 1
}

public class AgentSoccer : Agent
{
    // Note that that the detectable tags are different for the blue and purple teams. The order is
    // * ball
    // * own goal
    // * opposing goal
    // * wall
    // * own teammate
    // * opposing player

    public enum Position
    {
        Striker,
        Goalie,
        Generic
    }

    public GameObject ball;
    public GameObject GoalBlue;
    public GameObject GoalPurple;

    [HideInInspector]
    public Rigidbody ballRigdbody;   

    [HideInInspector]
    public Team team;
    float m_KickPower;
    // The coefficient for the reward for colliding with a ball. Set using curriculum.
    float m_BallTouch;
    public Position position;

    const float k_Power = 2000f;
    float m_Existential;
    float m_LateralSpeed;
    float m_ForwardSpeed;


    [HideInInspector]
    public Rigidbody agentRb;
    SoccerSettings m_SoccerSettings;
    BehaviorParameters m_BehaviorParameters;
    public Vector3 initialPos;
    public float rotSign;

    EnvironmentParameters m_ResetParams;

    public override void Initialize()
    {
        SoccerEnvController envController = GetComponentInParent<SoccerEnvController>();
        if (envController != null)
        {
            m_Existential = 1f / envController.MaxEnvironmentSteps;
        }
        else
        {
            m_Existential = 1f / MaxStep;
        }

        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();
        if (m_BehaviorParameters.TeamId == (int)Team.Blue)
        {
            team = Team.Blue;
            initialPos = new Vector3(transform.position.x - 5f, .5f, transform.position.z);
            rotSign = 1f;
        }
        else
        {
            team = Team.Purple;
            initialPos = new Vector3(transform.position.x + 5f, .5f, transform.position.z);
            rotSign = -1f;
        }
        if (position == Position.Goalie)
        {
            m_LateralSpeed = 1.0f;
            m_ForwardSpeed = 1.0f;
        }
        else if (position == Position.Striker)
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.3f;
        }
        else
        {
            m_LateralSpeed = 0.3f;
            m_ForwardSpeed = 1.0f;
        }
        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        agentRb = GetComponent<Rigidbody>();
        agentRb.maxAngularVelocity = 500;

        m_ResetParams = Academy.Instance.EnvironmentParameters;

        ballRigdbody = ball.GetComponent<Rigidbody>();

    }

    public void MoveAgent(ActionSegment<int> act)
    {
        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        m_KickPower = 0f;

        var forwardAxis = act[0];
        var rightAxis = act[1];
        var rotateAxis = act[2];

        switch (forwardAxis)
        {
            case 1:
                dirToGo = transform.forward * m_ForwardSpeed;
                m_KickPower = 1f;
                break;
            case 2:
                dirToGo = transform.forward * -m_ForwardSpeed;
                break;
        }

        switch (rightAxis)
        {
            case 1:
                dirToGo = transform.right * m_LateralSpeed;
                break;
            case 2:
                dirToGo = transform.right * -m_LateralSpeed;
                break;
        }

        switch (rotateAxis)
        {
            case 1:
                rotateDir = transform.up * -1f;
                break;
            case 2:
                rotateDir = transform.up * 1f;
                break;
        }

        transform.Rotate(rotateDir, Time.deltaTime * 100f);
        agentRb.AddForce(dirToGo * m_SoccerSettings.agentRunSpeed,
            ForceMode.VelocityChange);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)

    {

        if (position == Position.Goalie)
        {
            // Existential bonus for Goalies.
            AddReward(m_Existential);
        }
        else if (position == Position.Striker)
        {
            // Existential penalty for Strikers
            AddReward(-m_Existential);
        }
        MoveAgent(actionBuffers.DiscreteActions);
        //GiveReward();
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.DiscreteActions;
        //forward
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = 2;
        }
        //rotate
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[2] = 1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[2] = 2;
        }
        //right
        if (Input.GetKey(KeyCode.E))
        {
            discreteActionsOut[1] = 1;
        }
        if (Input.GetKey(KeyCode.Q))
        {
            discreteActionsOut[1] = 2;
        }
    }
    /// <summary>
    /// Used to provide a "kick" to the ball.
    /// </summary>
    void OnCollisionEnter(Collision c)
    {
        var force = k_Power * m_KickPower;
        if (position == Position.Goalie)
        {
            force = k_Power;
        }
        if (c.gameObject.CompareTag("ball"))
        {
            AddReward(.2f * m_BallTouch);
            var dir = c.contacts[0].point - transform.position;
            dir = dir.normalized;
            c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force);
            c.gameObject.GetComponent<SoccerBallController>().previous_agent_touch_ball = this;
        }
    }

    public override void OnEpisodeBegin()
    {
        m_BallTouch = m_ResetParams.GetWithDefault("ball_touch", 0);
    }

    void GiveReward()
    {
        float halfRewardDistance = 16;
        float goalieRewardDstance = 7;

        switch (position)
        {
            case Position.Striker:
                if (team == Team.Blue && Vector3.Distance(GoalPurple.transform.position, ball.transform.position) < halfRewardDistance)
                {
                    AddReward(0.002f);
                }
                else if (team == Team.Purple && Vector3.Distance(GoalBlue.transform.position, ball.transform.position) < halfRewardDistance)
                {
                    AddReward(0.002f);
                }
                else if (team == Team.Blue && Vector3.Distance(GoalBlue.transform.position, ball.transform.position) < halfRewardDistance)
                {
                    AddReward(-0.002f);
                }
                else if (team == Team.Purple && Vector3.Distance(GoalPurple.transform.position, ball.transform.position) < halfRewardDistance)
                {
                    AddReward(-0.002f);
                }
                break;
            case Position.Goalie:
                if(team == Team.Blue)
                {
                    if (Vector3.Distance(transform.position, GoalBlue.transform.position) >= goalieRewardDstance)
                    {
                        AddReward(-(Vector3.Distance(transform.position, GoalBlue.transform.position) - goalieRewardDstance) / 1000);
                    }
                    else if (Vector3.Distance(ball.transform.position, GoalBlue.transform.position) <= goalieRewardDstance * 0.7)
                    {
                        Vector3 blueGoalToBall = ball.transform.position - GoalBlue.transform.position;
                        blueGoalToBall.y = 0;
                        var r = Vector3.Dot(ballRigdbody.velocity, blueGoalToBall) / 1000;                      
                        AddReward(r);
                    }

                }
                else if (team == Team.Purple)
                {
                    if (Vector3.Distance(transform.position, GoalPurple.transform.position) >= goalieRewardDstance)
                    {
                        AddReward(-(Vector3.Distance(transform.position, GoalPurple.transform.position) - goalieRewardDstance) / 1000);
                    }
                    else if (Vector3.Distance(ball.transform.position, GoalPurple.transform.position) <= goalieRewardDstance * 0.7)
                    {
                        Vector3 blueGoalToBall = ball.transform.position - GoalPurple.transform.position;
                        blueGoalToBall.y = 0;
                        var r = Vector3.Dot(ballRigdbody.velocity, blueGoalToBall) / 1000;
                        if (r < 0)
                        {
                            r = r * 0.7f;
                        }
                        AddReward(r);
                    }

                }
                break;
            case Position.Generic:
                break;
        }
    }


}
