subroutine compute(N, verlet_list, distance_list, rc, particleClusters, cluster, max_neigh)

    use deque_mod
    
    implicit none
    type(deque):: toProcess
    integer(8), intent(in) :: N, max_neigh
    integer(8), dimension(0:N-1, 0:max_neigh-1), intent(in) :: verlet_list 
    real(8), dimension(0:N-1, 0:max_neigh-1), intent(in) :: distance_list
    real(8) , intent(in) :: rc 
    integer(8), dimension(0:N-1), intent(out) :: particleClusters
    !f2py intent(out) :: particleCluster 
    integer(8), intent(out) :: cluster
    !f2py intent(out) :: cluster

    integer(8) :: seedParticleIndex, currentParticle, n1, j, neighborIndex

    particleClusters = -1 
    cluster = 0
    do seedParticleIndex = 0, N-1
        if (particleClusters(seedParticleIndex) /= -1) then
            cycle
        end if
        call init(toProcess)
        call append(toProcess, seedParticleIndex)
        cluster = cluster + 1

        do while (remain(toProcess))
            currentParticle = popleft(toProcess)
            n1 = 0  ! 领域计数,保证孤立原子也有cluster_id
            do j = 0, max_neigh-1
                neighborIndex = verlet_list(currentParticle, j)
                if ((neighborIndex > -1) .and. (distance_list(currentParticle, j) <= rc)) then
                    n1 = n1 + 1
                    if (particleClusters(neighborIndex) == -1) then
                        particleClusters(neighborIndex) = cluster
                        call append(toProcess, neighborIndex)
                    end if
                end if
            end do
            if (n1 == 0) then
                particleClusters(currentParticle) = cluster
            end if
        end do 

    end do

end subroutine
