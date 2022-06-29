using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class pos : MonoBehaviour
{
    // Start is called before the first frame update
    private Vector3 position;
    private Vector3 rotaion;

    void Start()
    {
        position = new Vector3(0, 0, 0);
        rotaion = new Vector3(0, 0, 0);

        // transform.SetPositionAndRotation(new Vector3(0, 0, 1), Quaternion.Euler(new Vector3(0, 0, 0)));


    }

    public void setPose(Vector3 a, Vector3 b)
    {
        position = a;
        rotaion = b;
    }

    public void setX(float a)
    {
        position.x = a;
    }
    public void sety(float a)
    {
        position.y = a;
    }
    public void setz(float a)
    {
        position.z = a;
    }
    
    public void set_rotaion_X(float a)
    {
        rotaion.x = a;
    }
    public void set_rotaion_y(float a)
    {
        rotaion.y = a;
    }
    public void set_rotaion_z(float a)
    {
        rotaion.z = a;
    }
    
    // Update is called once per frame
    void Update()
    {
        transform.SetPositionAndRotation(position, Quaternion.Euler(rotaion));


    }
}