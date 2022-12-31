using UnityEngine;



public class SoccerBallController : MonoBehaviour
{
    public GameObject area;
    [HideInInspector]
    public SoccerEnvController envController;
    public string purpleGoalTag; //will be used to check if collided with purple goal
    public string blueGoalTag; //will be used to check if collided with blue goal

    public AgentSoccer previous_agent_touch_ball;

    

    void Start()
    {
        envController = area.GetComponent<SoccerEnvController>();
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag(purpleGoalTag)) //ball touched purple goal
        {
            if (previous_agent_touch_ball.team == Team.Blue) //蓝队踢进了紫队球门
            {
                previous_agent_touch_ball.AddReward(1.0f);
            }
            if (previous_agent_touch_ball.team == Team.Purple) //紫队踢进了紫队球门，或者没防守住
            {
                previous_agent_touch_ball.AddReward(-1.0f);
            }                 
            envController.GoalTouched(Team.Blue);        
        }
        if (col.gameObject.CompareTag(blueGoalTag)) //ball touched blue goal
        {
            if (previous_agent_touch_ball.team == Team.Purple) //紫队踢进了蓝队球门
            {
                previous_agent_touch_ball.AddReward(1.0f);
            }
            if (previous_agent_touch_ball.team == Team.Blue) //蓝队踢进了蓝队球门，或者没防守住
            {
                previous_agent_touch_ball.AddReward(-1.0f);
            }                
            envController.GoalTouched(Team.Purple);   
        }
    }
}
