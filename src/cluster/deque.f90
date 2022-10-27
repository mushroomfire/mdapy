module deque_mod
    implicit none 
    type deque
        integer(8),pointer:: v(:)
        integer(8):: l,r
        integer(8):: lmax, rmax
    end type
    private
    public:: deque
    public:: init
    public:: append, appendleft
    public:: pop, popleft
    public:: right, left
    public:: remaining_elements
    public:: remain

contains
    subroutine init(dq)
        type(deque):: dq
        dq%r=0; dq%rmax=1
        dq%l=1; dq%lmax=-1
        allocate(dq%v(dq%lmax:dq%rmax))
    end subroutine


    subroutine append(dq,num)
        type(deque):: dq
        integer(8):: num
        if (dq%r+1 > dq%rmax) call add_(dq)
        dq%r=dq%r+1
        dq%v(dq%r) = num
    end subroutine

    subroutine appendleft(dq,num)
        type(deque):: dq
        integer(8):: num
        if (dq%l-1 < dq%lmax) call add_(dq)
        dq%l=dq%l-1
        dq%v(dq%l) = num
    end subroutine

    subroutine add_(dq)
        type(deque):: dq
        integer(8):: l
        integer(8),pointer:: tmp(:)
        l = size(dq%v)
        allocate(tmp(l))
        tmp(:) = dq%v(:)
        deallocate(dq%v)
        allocate(dq%v(2*dq%lmax:2*dq%rmax))
        dq%v(dq%lmax:dq%rmax) = tmp(:)
        dq%lmax = 2*dq%lmax
        dq%rmax = 2*dq%rmax
    end subroutine


    function pop(dq) result(ret)
        type(deque):: dq
        integer(8):: ret
        ret = dq%v(dq%r)
        dq%r=dq%r-1
    end function

    function popleft(dq) result(ret)
        type(deque):: dq
        integer(8):: ret
        ret = dq%v(dq%l)
        dq%l=dq%l+1
    end function


    function right(dq) result(ret)
        type(deque):: dq
        integer(8):: ret
        ret = dq%v(dq%r)
    end function

    function left(dq) result(ret)
        type(deque):: dq
        integer(8):: ret
        ret = dq%v(dq%l)
    end function

    function remaining_elements(dq) result(ret)
        type(deque):: dq
        integer(8):: ret 
        ret = dq%r - dq%l + 1
    end function

    function remain(dq) result(ret)
        type(deque):: dq
        logical:: ret
        ret = remaining_elements(dq) > 0
    end function
end module