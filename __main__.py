import atexit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import Modules.cubic_spline_planner as csp
import Modules.lqr_control as lqr
from Modules.Turtle import turtle
import numpy as np
import subprocess
import time


def compute_traj(goal, ds=0.05):
    cx, cy, cyaw = csp.calc_spline_course([point[0] for point in goal], [
        point[1] for point in goal], ds)
    desired_traj = np.array([cx, cy, cyaw]).T
    return desired_traj


def reset_encoder():
    subprocess.call("./../../reset.sh")
    time.sleep(2)


def goodbye():
    # Stop the turtle
    turtle.set_velocity(0, 0)


if __name__ == '__main__':
    print("LQR steering control tracking start")

    # Add exit function
    atexit.register(goodbye)

    # Create the track waypoints
    goal = [(4.4731431876293675, 1.5774656033118226), (4.795613661268719, 1.1716394740046268), (5.04687690247169, 0.8518507244612199), (5.187507609886764, 0.6887251917691464), (5.2372762693291115, 0.594971386825764), (5.249999999999999, 0.4735175940582004), (5.215298916352124, 0.3057956897601364), (5.11439181784975, 0.25), (4.990624619505661, 0.27249482527699986), (4.866884816753926, 0.44125167417508826), (4.736396566419091, 0.6377693899914768), (4.680871788627784, 0.7025447461341774), (4.584859064897113, 0.7453427492998903), (4.512516741750882, 0.744977474735176), (4.4263819554365025, 0.7592231827590404), (4.351190186290028, 0.8309691951783756), (4.272860099841714, 1.0838913916960915), (4.20434067941069, 1.1808048216242542), (4.091501278460976, 1.221228540119323), (3.914312675027395, 1.2277791306465358), (3.8153841470838907, 1.2819950079142821), (3.7716425179593323, 1.3819219530013391), (3.7858577864361376, 1.4886694873980273), (3.842170948496286, 1.651643735541215), (3.8277121636430045, 1.8250943625958844), (3.719286497016924, 1.9527882625106536), (3.5210641665652007, 2.0311092170948495), (3.324181176184098, 2.0322963594301715), (3.1964872762693286, 1.9793011079995129), (3.1001461098258853, 1.899793011079995), (2.9411299159868496, 1.6974187264093508), (2.8688664312675023, 1.6082765128454888), (2.8013819554365025, 1.5143187629368073), (2.7558139534883717, 1.4537114939729696), (2.682363326433702, 1.3703214416169487), (2.5691890904663337, 1.330613052477779), (2.467247047363935, 1.3393400706197491), (2.362200170461463, 1.4079812492390111), (2.315110191160355, 1.5166504322415681), (2.286618775112626, 1.657993425057835), (2.26649823450627, 1.7421435529039326), (2.192834530622184, 2.079599415560696), (2.175423109704127, 2.2149640813344695), (2.202818702057713, 2.355320832826007), (2.2559357116766097, 2.4729392426640686),
            (2.2768781200535733, 2.554791184707171), (2.2511871423353216, 2.594301716790454), (2.188999147692682, 2.596280287349324), (2.1460185072446114, 2.5389017411420918), (2.100115670278826, 2.452088152928284), (2.033635699500791, 2.331243151101911), (2.004657250700109, 2.2808961402654324), (1.9476439790575912, 2.1991659564105683), (1.854133690490685, 2.1185315962498477), (1.7229696822111285, 2.0776512845488857), (1.4596676001461093, 2.0771946913429926), (1.2468951661999266, 2.120601485449896), (1.0837391939607937, 2.199409472787045), (0.920887617192256, 2.3446974309022277), (0.8246986484841101, 2.507792524047242), (0.7251613295994153, 2.7239133081699745), (0.6253196152441252, 2.888165104103251), (0.4588152928284419, 2.9067027882625105), (0.3471021551199316, 2.873249726044076), (0.25, 2.7738950444417387), (0.27009010105929576, 2.629702910020699), (0.35319006453184, 2.534244490441982), (0.5830086448313647, 2.3541032509436257), (0.769298672835748, 2.237337148423231), (1.1038292950200894, 2.073937659807622), (1.319341288201631, 2.0001521977352974), (1.7774869109947642, 1.8384268842079627), (2.286618775112626, 1.657993425057835), (2.4123341044685254, 1.6287562401071471), (2.5416108608303905, 1.6253257031535369), (2.69374771703397, 1.6996803847558746), (2.8583039084378417, 1.8574516011201752), (2.991750882746864, 1.9934859369292584), (3.1348471934737607, 2.132533787897236), (3.246864726652867, 2.219895287958115), (3.363417752343844, 2.27642152684768), (3.3763849993912087, 2.2459819797881404), (3.409411907950809, 2.189942773651528), (3.480914403993668, 2.2162729818580296), (3.5797820528430533, 2.2602885669061243), (3.748356264458784, 2.276969438694752), (3.8663399488615604, 2.2299707780348226), (3.9931815414586627, 2.135395105320833), (4.268568123706318, 1.8090831608425664), (4.4731431876293675, 1.5774656033118226)]
    # goal = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0), (1.5, 0.0), (2.0, 0.0), (2.5, 1.0), (0.5, 2.0), (0.0, 4.0), (1.0, 4.0), (2.0, 2.0), (0.0, 0.0)]

    # Scale the map
    factor = 4
    goal = [(pos[0] * factor, pos[1] * factor) for pos in goal]

    # Compute the desired trajectory
    desired_traj = compute_traj(goal)

    # Test on robot
    # '''
    # Reset the encoders
    reset_encoder()

    start = time.time()
    # Calculate the trajectory
    trajectory = lqr.closed_loop_prediction(desired_traj)
    end = time.time()
    print("Elapsed time: ", round(end - start, 2))
    np.savetxt("Robot.csv", trajectory, delimiter=",")


    # Stop the turtle
    turtle.set_velocity(0, 0)

    # Display the trajectory that the mobile robot executed
    plt.close()
    flg, ax = plt.subplots(1)
    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [
        exit(0) if event.key == 'escape' else None])
    plt.plot(desired_traj[:, 0], desired_traj[:, 1], "-b", label="Trajectory")
    plt.plot(trajectory[:, 0], trajectory[:, 1], "--g", label="Tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.xlim(0, 22.5)
    plt.ylim(0, 12.5)
    ax.xaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    plt.legend()
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("Robot.svg")
    plt.show()
    # '''

    # Simulation on robot
    '''
    # Calculate the trajectory
    trajectory = lqr.closed_loop_prediction(desired_traj, True)
    np.savetxt("Simulering.csv", trajectory, delimiter=",")

    # Display the trajectory that the mobile robot executed
    plt.close()
    flg, ax = plt.subplots(1)
    plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [
        exit(0) if event.key == 'escape' else None])
    plt.plot(desired_traj[:, 0], desired_traj[:, 1], ".b", label="Trajectory")
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-g", label="Tracking")
    plt.grid(True)
    plt.axis("equal")
    plt.xlabel("x[m]")
    plt.ylabel("y[m]")
    plt.xlim(0, 22.5)
    plt.ylim(0, 12.5)
    ax.xaxis.set_major_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_locator(MultipleLocator(2.5))
    plt.legend()
    plt.gca().set_position([0, 0, 1, 1])
    plt.savefig("Simulering.svg")
    plt.show()
    # '''
