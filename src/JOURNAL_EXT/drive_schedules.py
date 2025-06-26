import time
import argparse
import math
from pca9685 import PCA9685

class Ordinary_Car:
    def __init__(self):
        self.pwm = PCA9685(0x40, debug=True)
        self.pwm.set_pwm_freq(50)
    def duty_range(self, duty1, duty2, duty3, duty4):
        if duty1 > 4095:
            duty1 = 4095
        elif duty1 < -4095:
            duty1 = -4095        
        if duty2 > 4095:
            duty2 = 4095
        elif duty2 < -4095:
            duty2 = -4095  
        if duty3 > 4095:
            duty3 = 4095
        elif duty3 < -4095:
            duty3 = -4095
        if duty4 > 4095:
            duty4 = 4095
        elif duty4 < -4095:
            duty4 = -4095
        return duty1,duty2,duty3,duty4
    def left_upper_wheel(self,duty):
        if duty>0:
            self.pwm.set_motor_pwm(0,0)
            self.pwm.set_motor_pwm(1,duty)
        elif duty<0:
            self.pwm.set_motor_pwm(1,0)
            self.pwm.set_motor_pwm(0,abs(duty))
        else:
            self.pwm.set_motor_pwm(0,4095)
            self.pwm.set_motor_pwm(1,4095)
    def left_lower_wheel(self,duty):
        if duty>0:
            self.pwm.set_motor_pwm(3,0)
            self.pwm.set_motor_pwm(2,duty)
        elif duty<0:
            self.pwm.set_motor_pwm(2,0)
            self.pwm.set_motor_pwm(3,abs(duty))
        else:
            self.pwm.set_motor_pwm(2,4095)
            self.pwm.set_motor_pwm(3,4095)
    def right_upper_wheel(self,duty):
        if duty>0:
            self.pwm.set_motor_pwm(6,0)
            self.pwm.set_motor_pwm(7,duty)
        elif duty<0:
            self.pwm.set_motor_pwm(7,0)
            self.pwm.set_motor_pwm(6,abs(duty))
        else:
            self.pwm.set_motor_pwm(6,4095)
            self.pwm.set_motor_pwm(7,4095)
    def right_lower_wheel(self,duty):
        if duty>0:
            self.pwm.set_motor_pwm(4,0)
            self.pwm.set_motor_pwm(5,duty)
        elif duty<0:
            self.pwm.set_motor_pwm(5,0)
            self.pwm.set_motor_pwm(4,abs(duty))
        else:
            self.pwm.set_motor_pwm(4,4095)
            self.pwm.set_motor_pwm(5,4095)
    def set_motor_model(self, duty1, duty2, duty3, duty4):
        duty1,duty2,duty3,duty4=self.duty_range(duty1,duty2,duty3,duty4)
        self.left_upper_wheel(duty1)
        self.left_lower_wheel(duty2)
        self.right_upper_wheel(duty3)
        self.right_lower_wheel(duty4)

    def close(self):
        self.set_motor_model(0,0,0,0)
        self.pwm.close()

class Robot():
    def __init__(self, pwm):
        self.pwm = pwm
        
    def set_speed(self, speed):
        duty_speed = speed / 100 * 4095
        duty_speed = int(duty_speed)
        self.speed = speed
        self.duty_speed = duty_speed
    
    def forward(self, run_time):
        self.pwm.set_motor_model(self.duty_speed, self.duty_speed, self.duty_speed, self.duty_speed)
        time.sleep(run_time)
        
    def backward(self, run_time):
        self.pwm.set_motor_model(-self.duty_speed, -self.duty_speed, -self.duty_speed, -self.duty_speed)
        time.sleep(run_time)
        
    # Right rotate
    def rotate_left(self, run_time):
        self.pwm.set_motor_model(self.duty_speed, self.duty_speed, -self.duty_speed, -self.duty_speed)
        time.sleep(run_time)
        
    # Left rotate
    def rotate_right(self, run_time):
        self.pwm.set_motor_model(-self.duty_speed, -self.duty_speed, self.duty_speed, self.duty_speed)
        time.sleep(run_time)
        
    def strafe_right(self, run_time):
        self.pwm.set_motor_model(self.duty_speed, -self.duty_speed, -self.duty_speed, self.duty_speed)
        time.sleep(run_time)
        
    def strafe_left(self, run_time):
        self.pwm.set_motor_model(-self.duty_speed, self.duty_speed, self.duty_speed, -self.duty_speed)
        time.sleep(run_time)
        
    def diagonal_forward_right(self, run_time):
        """Moves diagonally forward and to the right."""
        self.pwm.set_motor_model(self.duty_speed, 0, 0, self.duty_speed)
        time.sleep(run_time)
    
    def diagonal_forward_left(self, run_time):
        """Moves diagonally forward and to the left."""
        self.pwm.set_motor_model(0, self.duty_speed, self.duty_speed, 0)
        time.sleep(run_time)
        
    def diagonal_backward_left(self, run_time):
        """Moves diagonally backward and to the left."""
        self.pwm.set_motor_model(-self.duty_speed, 0, 0, -self.duty_speed)
        time.sleep(run_time)
        
    def diagonal_backward_right(self, run_time):
        """Moves diagonally backward and to the right."""
        self.pwm.set_motor_model(0, -self.duty_speed, -self.duty_speed, 0)
        time.sleep(run_time)
        
    def stop(self):
        self.pwm.set_motor_model(0, 0, 0, 0)
        
# ============================================================================
#
#  MovementScheduler Class
#  (New class to run complex, timed movement patterns)
#
# ============================================================================

class MovementScheduler:
    """
    Runs complex, timed movement sequences (schedules) for the robot.
    """
    def __init__(self, robot):
        self.robot = robot

    def run_schedule(self, schedule_name, total_duration, speed, step_time=1.0):
        """
        Runs a given schedule for a total duration.
        :param schedule_name: The name of the schedule to execute.
        :param total_duration: The total time in seconds the schedule should run.
        :param speed: The speed percentage (0-100) for the robot.
        :param step_time: The duration of a single step within a repeating pattern.
        """
        self.robot.set_speed(speed)
        
        schedules = {
            'forward_backward': self._get_forward_backward_steps,
            'square_strafe': self._get_square_strafe_steps,
            'square_rotate': self._get_square_rotate_steps,
            'diamond': self._get_diamond_steps,
            'integer_sine': self._get_integer_sine_steps,
            'zigzag': self._get_zigzag_steps,
            'x_path': self._get_x_path_steps
        }

        schedule_func = schedules.get(schedule_name)
        if not schedule_func:
            print(f"Error: Schedule '{schedule_name}' not found.")
            return
            
        print(f"Running schedule '{schedule_name}' for {total_duration}s at {speed}% speed.")
        
        # Get the list of moves for the selected schedule
        steps = schedule_func(step_time)
        
        start_time = time.time()
        current_step_index = 0
        
        while time.time() - start_time < total_duration:
            elapsed_time = time.time() - start_time
            remaining_time = total_duration - elapsed_time
            
            # Get the current move and its intended duration
            move_function, move_duration = steps[current_step_index]
            
            # If the remaining time is less than a full step, only run for the remaining time.
            time_to_run = min(remaining_time, move_duration)
            
            # Execute the move
            move_function(time_to_run)
            
            # If the loop was cut short, break out.
            if time_to_run < move_duration:
                break
                
            # Move to the next step in the sequence
            current_step_index = (current_step_index + 1) % len(steps)
            
        print("Schedule complete.")
        self.robot.stop()

    # --- Methods to define the steps for each schedule ---

    def _get_forward_backward_steps(self, n):
        return [
            (self.robot.forward, n),
            (self.robot.backward, n)
        ]

    def _get_square_strafe_steps(self, step_time):
        return [
            (self.robot.forward, step_time),
            (self.robot.strafe_right, step_time),
            (self.robot.backward, step_time),
            (self.robot.strafe_left, step_time)
        ]

    def _get_square_rotate_steps(self, step_time):
        rotation_time = step_time / 2.0 # Rotations are usually quicker
        return [
            (self.robot.forward, step_time),
            (self.robot.rotate_right, rotation_time),
            (self.robot.forward, step_time),
            (self.robot.rotate_right, rotation_time),
            (self.robot.forward, step_time),
            (self.robot.rotate_right, rotation_time),
            (self.robot.forward, step_time),
            (self.robot.rotate_right, rotation_time)
        ]

    def _get_diamond_steps(self, step_time):
        return [
            (self.robot.diagonal_forward_right, step_time),
            (self.robot.diagonal_backward_right, step_time),
            (self.robot.diagonal_backward_left, step_time),
            (self.robot.diagonal_forward_left, step_time)
        ]

    def _get_integer_sine_steps(self, step_time):
        return [
            (self.robot.forward, step_time),
            (self.robot.strafe_right, step_time),
            (self.robot.forward, step_time),
            (self.robot.strafe_left, step_time),
            (self.robot.forward, step_time),
            (self.robot.backward, step_time),
            (self.robot.strafe_right, step_time),
            (self.robot.backward, step_time),
            (self.robot.strafe_left, step_time),
            (self.robot.backward, step_time)
        ]

    def _get_zigzag_steps(self, step_time):
        return [
            (self.robot.diagonal_forward_right, step_time),
            (self.robot.diagonal_forward_left, step_time)
        ]

    def _get_x_path_steps(self, step_time):
        """Returns the steps for the 8-part 'X' pattern."""
        return [
            (self.robot.diagonal_forward_right, step_time),
            (self.robot.diagonal_backward_left, step_time),
            (self.robot.diagonal_forward_left, step_time),
            (self.robot.diagonal_backward_right, step_time),
            (self.robot.diagonal_backward_left, step_time),
            (self.robot.diagonal_forward_right, step_time),
            (self.robot.diagonal_backward_right, step_time),
            (self.robot.diagonal_forward_left, step_time),
        ]

if __name__ == '__main__':
    motor_controller = Ordinary_Car()
    robot = Robot(motor_controller)
    scheduler = MovementScheduler(robot)
    
    # --- Command-line argument setup ---
    parser = argparse.ArgumentParser(description="Run movement schedules for a mecanum wheel robot.")
    parser.add_argument(
        '-s', '--schedule',
        type=str,
        default='forward_backward',
        help="The name of the schedule to run. Available: forward_backward, square_strafe, square_rotate, diamond, integer_sine, zigzag, x_path"
    )
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=5,
        help="Total execution time for the schedule in seconds."
    )
    parser.add_argument(
        '--speed',
        type=int,
        default=20,
        help="The speed of the robot as a percentage (0-100)."
    )
    parser.add_argument(
        '-t', '--steptime',
        type=float,
        default=1.0,
        help="The duration of a single step in a schedule pattern (e.g., 'n')."
    )
    
    args = parser.parse_args()

    try:
        # Run the schedule using the parsed arguments
        scheduler.run_schedule(args.schedule, args.duration, args.speed, args.steptime)

    except KeyboardInterrupt:
        print ("\nProgram interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        motor_controller.close()
